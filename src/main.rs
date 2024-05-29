use anyhow::{Error as E, Result};
use candle_core::Device;
use colored::Colorize;
use cs470::cmd_args::parse_args;
use cs470::hf_models::t5::T5Model;
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
        Device::new_cuda(0).map_err(E::msg)?
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

    for (title, report, filename) in [
        ("Draft only", reports.0, "draft.timings"),
        ("Target only", reports.1, "target.timings"),
        ("Speculative sampling", reports.2, "spec.timings"),
    ] {
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
        info!(
            "Generated text : {}\n",
            tokenizer
                .decode(&report.output_tokens, true)
                .map_err(E::msg)?
                .cyan()
        );
        report.export_timings(filename)?;
    }
    Ok(())
}
