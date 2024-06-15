use super::report::{ActionType, RunnerType, TaskReport};
use crate::hf_models::t5::T5Model;
use anyhow::Result;
use candle_core::Tensor;

pub fn sampling(
    runner_type: RunnerType,
    model: &mut T5Model,
    tokens: &[u32],
    max_tokens: usize,
) -> Result<TaskReport> {
    let mut report = TaskReport::new();
    report.output_tokens.push(
        model
            .config
            .decoder_start_token_id
            .unwrap_or(model.config.pad_token_id) as u32,
    );

    let input_tokens = Tensor::new(tokens, &model.device)?.unsqueeze(0)?;
    model.runner.clear_kv_cache();
    model.reset_rng();
    let encoder_output = model.runner.encode(&input_tokens)?;
    for i in 0..max_tokens {
        report.start(runner_type, ActionType::ForwardKV, i);
        let decoder_output = model.runner.forward_kv_cache(
            i..i + 1,
            &encoder_output,
            report.output_tokens.as_slice(),
        )?;
        report.end();
        report.start(runner_type, ActionType::LogitsCalc, i);
        let logits = model.runner.get_logits(decoder_output)?;
        report.end();
        report.start(runner_type, ActionType::Sampling, i);
        let p = model.p_from_logits(&logits, 0, report.output_tokens.as_slice())?;
        let next_token = model.sample_from_p(&p)?;
        report.end();
        if next_token as usize == model.config.eos_token_id {
            break;
        }
        report.output_tokens.push(next_token);
    }
    Ok(report)
}
