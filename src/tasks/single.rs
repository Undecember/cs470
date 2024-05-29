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
    let mut output_tokens = [model
        .config
        .decoder_start_token_id
        .unwrap_or(model.config.pad_token_id) as u32]
    .to_vec();

    let input_tokens = Tensor::new(tokens, &model.device)?.unsqueeze(0)?;
    let encoder_output = model.runner.write().unwrap().encode(&input_tokens)?;
    for i in 0..max_tokens {
        let span = report.start(runner_type, ActionType::ForwardKV, i);
        let decoder_output = model.runner.write().unwrap().forward_kv_cache(
            i..i + 1,
            &encoder_output,
            &output_tokens,
        )?;
        report.end(span);
        let span = report.start(runner_type, ActionType::LogitsCalc, i);
        let logits = model
            .runner
            .read()
            .unwrap()
            .get_logits(decoder_output, &output_tokens)?;
        report.end(span);
        let span = report.start(runner_type, ActionType::Sampling, i);
        let p = model.p_from_logits(&logits)?;
        let next_token = model.sample_from_p(&p)?;
        report.end(span);
        if next_token as usize == model.config.eos_token_id {
            break;
        }
        output_tokens.push(next_token);
    }
    report.set_output_tokens(output_tokens.as_slice());
    Ok(report)
}
