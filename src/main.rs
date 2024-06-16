use anyhow::{Error as E, Result};
use candle_core::Device;
use colored::Colorize;
use cs470::cmd_args::parse_args;
use cs470::hf_models::T5Model;
use cs470::tasks::run_exp;
use log::info;
use std::sync::Arc;

fn main() -> Result<()> {
    let args = &parse_args();

    std::env::set_var("RUST_LOG", if args.quiet { "off" } else { "trace" });
    env_logger::init();

    args.review();

    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_cuda(0)?
    };
    let device = Arc::new(device);

    info!("Loading draft model...");
    let (mut draft_model, _) =
        T5Model::new(args.get_draft_repo(), device.clone(), args.get_model_args())?;
    info!("Loading target model...");
    let (mut target_model, mut tokenizer) = T5Model::new(
        args.get_target_repo(),
        device.clone(),
        args.get_model_args(),
    )?;

    let prompt = format!(
        "{}{}",
        args.get_prefix(),
        if let Some(file) = &args.prompt_group.prompt_file {
            std::fs::read_to_string(file)?
        } else {
            args.prompt_group.prompt.as_ref().unwrap().clone()
        }
    );

    info!("");
    info!("Start experiment.\n");
    let reports = run_exp(
        args,
        &mut draft_model,
        &mut target_model,
        prompt,
        &mut tokenizer,
    )?;
    let (reports, kl_divs) = (reports.task_reports, reports.kl_divs);

    for (report, (title, filename)) in reports.iter().zip([
        ("Draft only", "draft.timings"),
        ("Target only", "target.timings"),
        ("Speculative sampling", "spec.timings"),
    ]) {
        info!("[ {} ]", title.bold());
        info!(
            "Generation speed : {:.3} ms/token ({:.3} ms / {} tokens in total)",
            report.total_millis() / report.output_tokens.len() as f64,
            report.total_millis(),
            report.output_tokens.len(),
        );
        if let Some(rate) = report.acceptance_rate() {
            info!("Acceptance rate : {:.3}", rate);
        }
        let result_text = tokenizer
            .decode(&report.output_tokens, true)
            .map_err(E::msg)?;
        info!("Generated text : {}\n", (result_text).cyan());
        report.export_timings(filename)?;
    }
    info!("[ {} ]", "Statistics".bold());
    info!(
        "Average KL divergence for accepted tokens : {:.5} (total {} tokens)",
        kl_divs.0.iter().sum::<f64>() / kl_divs.0.len() as f64,
        kl_divs.0.len(),
    );
    info!(
        "Average KL divergence for rejected tokens : {:.5} (total {} tokens)",
        kl_divs.1.iter().sum::<f64>() / kl_divs.1.len() as f64,
        kl_divs.1.len(),
    );
    Ok(())
}
