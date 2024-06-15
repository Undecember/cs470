mod cmd_args;
mod scenario;

use anyhow::{Error as E, Result};
use candle_core::Device;
use cmd_args::parse_args;
use cs470::hf_models::T5Model;
use cs470::tasks::run_exp;
use csv::Writer;
use std::sync::Arc;

fn main() -> Result<()> {
    let args = &parse_args();

    std::env::set_var("RUST_LOG", "off");
    env_logger::init();

    let scenario_config = std::fs::read_to_string(&args.scenario)?;
    let scenario_config: scenario::Config = serde_json::from_str(&scenario_config)?;
    let mut csv = Writer::from_path(&args.output)?;
    csv.write_record(&[
        "id",
        "gamma",
        "lenience",
        "k_skipping",
        "temperature",
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
    ])?;

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
        result.push(args.lenience.to_string());
        result.push(args.k_skipping.to_string());
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
        csv.write_record(result)?;
        progress_bar.inc(1);
    }

    progress_bar.finish_with_message("done");
    csv.flush()?;

    Ok(())
}
