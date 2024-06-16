mod cmd_args;
mod scenario;

use anyhow::{Error as E, Result};
use candle_core::Device;
use cmd_args::parse_args;
use cs470::hf_models::T5Model;
use cs470::tasks::run_exp;
use csv::{ReaderBuilder, WriterBuilder};
use std::sync::Arc;

const HEADERS: [&str; 25] = [
    "id",
    "gamma",
    "adaptive_gamma",
    "lenience",
    "k_skipping",
    "temperature",
    "early_reject_thr",
    "top_p",
    "kl_epsilon",
    "draft_gen_text",
    "draft_total_tokens",
    "draft_time_elapsed",
    "target_gen_text",
    "target_total_tokens",
    "target_time_elapsed",
    "spec_gen_text",
    "spec_total_tokens",
    "spec_time_elapsed",
    "accepted_tokens",
    "rejected_tokens",
    "kl_div_accepted",
    "kl_div_rejected",
    "bert_draft_target",
    "bert_draft_spec",
    "bert_target_spec",
];

fn main() -> Result<()> {
    let args = &parse_args();

    std::env::set_var("RUST_LOG", "off");
    env_logger::init();

    let scenario_config = std::fs::read_to_string(&args.scenario)?;
    let scenario_config: scenario::Config = serde_json::from_str(&scenario_config)?;
    let mut csv = WriterBuilder::new()
        .delimiter(b'\t')
        .from_path((args.output.clone() + ".without_bert").as_str())?;
    csv.write_record(&HEADERS)?;

    let device = if scenario_config.cpu {
        Device::Cpu
    } else {
        Device::new_cuda(0)?
    };
    let device = Arc::new(device);

    println!("Loading draft model...");
    let (mut draft_model, _) = T5Model::new(
        scenario_config.get_draft_repo()?,
        device.clone(),
        scenario_config.get_model_args(),
    )?;
    println!("Loading target model...");
    let (mut target_model, mut tokenizer) = T5Model::new(
        scenario_config.get_target_repo()?,
        device.clone(),
        scenario_config.get_model_args(),
    )?;

    let (progress_bar, iter) = scenario_config.iter()?;

    for (args, need_init) in iter {
        if need_init {
            draft_model.set_model_args(args.get_model_args());
            target_model.set_model_args(args.get_model_args());
        }
        let prompt = format!(
            "{}{}",
            args.get_prefix(),
            if let Some(file) = &args.prompt_group.prompt_file {
                std::fs::read_to_string(file)?
            } else {
                args.prompt_group.prompt.as_ref().unwrap().clone()
            }
        );
        let reports = run_exp(
            &args,
            &mut draft_model,
            &mut target_model,
            prompt,
            &mut tokenizer,
        )?;
        let (reports, kl_divs) = (reports.task_reports, reports.kl_divs);

        let mut result = Vec::with_capacity(14);
        result.push(progress_bar.position().to_string());
        result.push(args.gamma.to_string());
        result.push(args.adaptive_gamma.to_string());
        result.push(args.lenience.to_string());
        result.push(args.k_skipping.to_string());
        result.push(args.early_reject_thr.to_string());
        result.push(args.temperature.to_string());
        result.push(args.top_p.map_or("1.0".to_string(), |v| v.to_string()));
        result.push(args.kl_epsilon.to_string());

        for report in reports {
            result.push(
                tokenizer
                    .decode(&report.output_tokens, true)
                    .map_err(E::msg)?,
            );
            result.push(report.output_tokens.len().to_string());
            result.push(report.total_millis().to_string());
        }
        result.push(kl_divs.0.len().to_string());
        result.push(kl_divs.1.len().to_string());
        result.push(
            (kl_divs.0.iter().sum::<f64>() / kl_divs.0.len() as f64).to_string(),
        );
        result.push(
            (kl_divs.1.iter().sum::<f64>() / kl_divs.1.len() as f64).to_string(),
        );
        for _ in 0..3 {
            result.push("".to_string());
        }
        csv.write_record(result)?;
        progress_bar.inc(1);
    }

    progress_bar.finish_with_message("Batch speculation done");
    csv.flush()?;
    println!("csv generated. (without bert)");
    drop(csv);

    let mut read = ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path((args.output.clone() + ".without_bert").as_str())?;
    let mut write = WriterBuilder::new()
        .delimiter(b'\t')
        .from_path(args.output.as_str())?;

    progress_bar.reset();
    write.write_record(&HEADERS)?;
    for result in read.records() {
        let record = result?;
        let mut record: Vec<String> =
            record.into_iter().map(|s| s.to_string()).collect();
        record[22] =
            get_bert_score(record[9].as_str(), record[12].as_str())?.to_string();
        record[23] =
            get_bert_score(record[9].as_str(), record[15].as_str())?.to_string();
        record[24] =
            get_bert_score(record[12].as_str(), record[15].as_str())?.to_string();
        write.write_record(record)?;
        progress_bar.inc(1);
    }
    progress_bar.finish_with_message("Bert score evaluation done");
    write.flush()?;
    println!("csv generated.");

    Ok(())
}

fn get_bert_score(r: &str, c: &str) -> Result<f64> {
    use std::io::Write;
    std::fs::File::create("__r__")?.write_all(r.as_bytes())?;
    std::fs::File::create("__c__")?.write_all(c.as_bytes())?;
    let output = std::process::Command::new("bert-score")
        .arg("-m")
        .arg("microsoft/deberta-xlarge-mnli")
        .arg("-r")
        .arg("__r__")
        .arg("-c")
        .arg("__c__")
        .arg("--lang")
        .arg("en")
        .arg("--use_fast_tokenizer")
        .stdout(std::process::Stdio::piped())
        .output()?;
    let output = String::from_utf8(output.stdout).map_err(E::msg)?;
    let start = output
        .find("F1: ")
        .ok_or(E::msg("Can't find F1 value from output."))?;
    let f1_str = &output[start + 4..];
    f1_str.trim().parse::<f64>().map_err(E::msg)
}
