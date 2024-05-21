use crate::t5_model::T5Model;
use anyhow::Result;
use candle_core::Tensor;
use std::time::Instant;

pub enum TimingsReportItem {
    DraftBegin(usize),
    DraftEnd(usize),
    TargetBegin(usize),
    TargetEnd(usize),
}

pub struct SamplingResult {
    pub output_tokens: Vec<u32>,
    pub timings_report: Vec<(Instant, TimingsReportItem)>,
}

impl Default for SamplingResult {
    fn default() -> Self {
        Self::new()
    }
}

impl SamplingResult {
    pub fn new() -> Self {
        Self {
            output_tokens: Vec::<u32>::new(),
            timings_report: Vec::<(Instant, TimingsReportItem)>::new(),
        }
    }
}

pub fn single_sampling(
    model: &mut T5Model,
    tokens: &[u32],
    max_tokens: usize,
) -> Result<SamplingResult> {
    let mut result = SamplingResult::new();
    let mut output_tokens = [model
        .config
        .decoder_start_token_id
        .unwrap_or(model.config.pad_token_id) as u32]
    .to_vec();

    let input_tokens = Tensor::new(tokens, &model.device)?.unsqueeze(0)?;
    let encoder_output = model.cgs[0].encode(&input_tokens)?;
    for i in 0..max_tokens {
        let decoder_tokens = if i == 0 || !model.config.use_cache {
            Tensor::new(output_tokens.as_slice(), &model.device)?.unsqueeze(0)?
        } else {
            let last_token = *output_tokens.last().unwrap();
            Tensor::new(&[last_token], &model.device)?.unsqueeze(0)?
        };
        let begin_time = Instant::now();
        let logits =
            model.get_logits(0, &decoder_tokens, &encoder_output, &output_tokens)?;
        let p = model.p_from_logits(&logits)?;
        let next_token_id = model.sample_from_p(&p)?;
        let end_time = Instant::now();
        result
            .timings_report
            .push((begin_time, TimingsReportItem::TargetBegin(i)));
        result
            .timings_report
            .push((end_time, TimingsReportItem::TargetBegin(i)));
        if next_token_id as usize == model.config.eos_token_id {
            break;
        }
        output_tokens.push(next_token_id);
    }
    result.output_tokens = output_tokens;
    Ok(result)
}
