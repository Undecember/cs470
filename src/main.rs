use anyhow::{Error as E, Result};
use colored::Colorize;
use cs470::cmd_args::parse_args;
use cs470::t5::T5Model;
use cs470::tasks::single_sampling;
use log::info;

fn main() -> Result<()> {
    std::env::set_var("RUST_LOG", "trace");
    env_logger::init();
    let args = parse_args();
    args.review();

    info!("Loading draft model...");
    let (mut draft_model, _) =
        T5Model::new(args.get_draft_repo(), args.get_model_args())?;
    info!("Loading target model...");
    let (mut target_model, mut tokenizer) =
        T5Model::new(args.get_target_repo(), args.get_model_args())?;

    let prompt = format!("summarize: {}", args.prompt);
    let tokenizer = tokenizer
        .with_padding(None)
        .with_truncation(None)
        .map_err(E::msg)?;
    let tokens = tokenizer
        .encode(prompt.clone(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    info!("Start generating.");
    info!("[ {} ]\n", "Draft only".bold());
    let result = single_sampling(&mut draft_model, &tokens, args.max_tokens)?;
    let dur = result.timings_report[result.timings_report.len() - 1].0
        - result.timings_report[0].0;
    info!(
        "Generation speed : {:.3} ms/tokens",
        dur.as_millis() as f64 / result.output_tokens.len() as f64
    );
    info!(
        "Generated text :\n{}",
        tokenizer
            .decode(&result.output_tokens, true)
            .map_err(E::msg)?
            .cyan()
    );

    info!("[ {} ]\n", "Target only".bold());
    let result = single_sampling(&mut target_model, &tokens, args.max_tokens)?;
    let dur = result.timings_report[result.timings_report.len() - 1].0
        - result.timings_report[0].0;
    info!(
        "Generation speed : {:.3} ms/tokens",
        dur.as_millis() as f64 / result.output_tokens.len() as f64
    );
    info!(
        "Generated text :\n{}",
        tokenizer
            .decode(&result.output_tokens, true)
            .map_err(E::msg)?
            .cyan()
    );

    info!("[ {} ]\n", "Speculative sampling".bold());
    let result = speculative_sampling(
        &mut draft_model,
        &mut target_model,
        args.gamma,
        &tokens,
        args.max_tokens,
    )?;
    let dur = result.timings_report[result.timings_report.len() - 1].0
        - result.timings_report[0].0;
    info!(
        "Generation speed : {:.3} ms/tokens",
        dur.as_millis() as f64 / result.output_tokens.len() as f64
    );
    info!(
        "Generated text :\n{}",
        tokenizer
            .decode(&result.output_tokens, true)
            .map_err(E::msg)?
            .cyan()
    );
    info!("Timings report :\n{:?}", result.timings_report);

    Ok(())
}
