use super::builder::T5Model;
use anyhow::Result;
use candle_core::Tensor;
use candle_transformers::generation::LogitsProcessor;
use std::time::Instant;

pub enum TimingsReportItem {
    TokenBegin(u32),
    TokenEnd(u32),
}

pub struct SamplingResult {
    pub output_tokens: Vec<u32>,
    pub timings_report: Vec<(Instant, TimingsReportItem)>,
}

impl SamplingResult {
    pub fn new() -> Self {
        Self {
            output_tokens: Vec::<u32>::new(),
            timings_report: Vec::<(Instant, TimingsReportItem)>::new(),
        }
    }
}

impl T5Model {
    pub fn single_sampling(
        self: &mut Self,
        tokens: &Vec<u32>,
        max_tokens: usize,
    ) -> Result<SamplingResult> {
        let mut result = SamplingResult::new();
        let mut output_tokens = [self
            .config
            .decoder_start_token_id
            .unwrap_or(self.config.pad_token_id) as u32]
        .to_vec();
        let mut logits_processor =
            LogitsProcessor::new(self.seed, Some(self.temperature), self.top_p);

        let input_tokens = Tensor::new(&tokens[..], &self.device)?.unsqueeze(0)?;
        let encoder_output = self.model.encode(&input_tokens)?;
        for i in 0..max_tokens {
            let decoder_tokens = if i == 0 || !self.config.use_cache {
                Tensor::new(output_tokens.as_slice(), &self.device)?.unsqueeze(0)?
            } else {
                let last_token = *output_tokens.last().unwrap();
                Tensor::new(&[last_token], &self.device)?.unsqueeze(0)?
            };
            let begin_time = Instant::now();
            let logits = self
                .model
                .decode(&decoder_tokens, &encoder_output)?
                .squeeze(0)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = output_tokens.len().saturating_sub(64);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &output_tokens[start_at..],
                )?
            };
            let next_token_id = logits_processor.sample(&logits)?;
            let end_time = Instant::now();
            result
                .timings_report
                .push((begin_time, TimingsReportItem::TokenBegin(next_token_id)));
            result
                .timings_report
                .push((end_time, TimingsReportItem::TokenBegin(next_token_id)));
            if next_token_id as usize == self.config.eos_token_id {
                break;
            }
            output_tokens.push(next_token_id);
        }
        result.output_tokens = output_tokens;
        Ok(result)
    }
}
