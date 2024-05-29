mod report;
mod single;
mod speculative;

pub use report::RunnerType;
pub use report::TaskReport;
pub use single::sampling as single_sampling;
pub use speculative::sampling as speculative_sampling;

use crate::cmd_args::Args;
use crate::hf_models::t5::T5Model;
use anyhow::{Error as E, Result};
use tokenizers::Tokenizer;

pub struct ExpReport {
    pub task_reports: Vec<TaskReport>,
    pub kl_divs: (Vec<f64>, Vec<f64>),
}

pub fn run_exp(
    args: &Args,
    draft_model: &mut T5Model,
    target_model: &mut T5Model,
    prompt: String,
    tokenizer: &mut Tokenizer,
) -> Result<ExpReport> {
    let tokenizer = tokenizer
        .with_padding(None)
        .with_truncation(None)
        .map_err(E::msg)?;
    let tokens = tokenizer
        .encode(prompt.clone(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    let draft_report =
        single_sampling(RunnerType::Draft, draft_model, &tokens, args.max_tokens)?;
    draft_model.reset_rng();
    let target_report =
        single_sampling(RunnerType::Target, target_model, &tokens, args.max_tokens)?;
    target_model.reset_rng();
    let spec_report = speculative_sampling(
        draft_model,
        target_model,
        args.gamma,
        args.lenience,
        args.k_skipping,
        &tokens,
        args.max_tokens,
        None,
    )?;
    draft_model.reset_rng();
    target_model.reset_rng();
    let kl_report = speculative_sampling(
        draft_model,
        target_model,
        args.gamma,
        args.lenience,
        args.k_skipping,
        &tokens,
        args.max_tokens,
        Some(args.kl_epsilon),
    )?;
    Ok(ExpReport {
        task_reports: vec![draft_report, target_report, spec_report],
        kl_divs: kl_report.kl_divs.unwrap(),
    })
}
