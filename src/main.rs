use anyhow::{Error as E, Result};
use candle_core::Device;
use colored::Colorize;
use cs470::cmd_args::parse_args;
use cs470::hf_models::t5::T5Model;
use cs470::tasks::report::RunnerType::{Draft, Target};
use cs470::tasks::single::sampling as single_sampling;
use cs470::tasks::speculative::sampling as speculative_sampling;
use log::info;
use std::sync::Arc;

fn main() -> Result<()> {
    std::env::set_var("RUST_LOG", "trace");
    env_logger::init();
    let args = parse_args();
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
        if let Some(file) = args.prompt_group.prompt_file {
            std::fs::read_to_string(file)?
        } else {
            args.prompt_group.prompt.unwrap()
        }
    );
    let tokenizer = tokenizer
        .with_padding(None)
        .with_truncation(None)
        .map_err(E::msg)?;
    let tokens = tokenizer
        .encode(prompt.clone(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    info!("Start generating.\n");
    info!("[ {} ]", "Draft only".bold());
    let report = single_sampling(Draft, &mut draft_model, &tokens, args.max_tokens)?;
    let dur = report.total_millis();
    info!(
        "Generation speed : {:.3} ms/token",
        dur / report.output_tokens.len() as f64
    );
    info!(
        "Generated text : {}\n",
        tokenizer
            .decode(&report.output_tokens, true)
            .map_err(E::msg)?
            .cyan()
    );
    report.export_timings("draft.timings")?;

    info!("[ {} ]", "Target only".bold());
    let report = single_sampling(Target, &mut target_model, &tokens, args.max_tokens)?;
    let dur = report.total_millis();
    info!(
        "Generation speed : {:.3} ms/token",
        dur / report.output_tokens.len() as f64
    );
    info!(
        "Generated text : {}\n",
        tokenizer
            .decode(&report.output_tokens, true)
            .map_err(E::msg)?
            .cyan()
    );
    report.export_timings("target.timings")?;

    info!("[ {} ]", "Speculative sampling".bold());
    let report = speculative_sampling(
        draft_model,
        target_model,
        args.gamma,
        &tokens,
        args.max_tokens,
    )?;
    let dur = report.total_millis();
    info!(
        "Generation speed : {:.3} ms/token",
        dur / report.output_tokens.len() as f64
    );
    info!(
        "Generated text : {}\n",
        tokenizer
            .decode(&report.output_tokens, true)
            .map_err(E::msg)?
            .cyan()
    );
    report.export_timings("speculative.timings")?;

    Ok(())
}
