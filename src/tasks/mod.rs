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

pub fn run_exp(
    args: &Args,
    draft_model: &mut T5Model,
    target_model: &mut T5Model,
    prompt: String,
    tokenizer: &mut Tokenizer,
) -> Result<(TaskReport, TaskReport, TaskReport)> {
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
    let target_report =
        single_sampling(RunnerType::Target, target_model, &tokens, args.max_tokens)?;
    let spec_report = speculative_sampling(
        draft_model,
        target_model,
        args.gamma,
        &tokens,
        args.max_tokens,
    )?;
    Ok((draft_report, target_report, spec_report))
}
