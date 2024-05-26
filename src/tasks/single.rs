use crate::t5::T5Model;
use anyhow::Result;
use candle_core::Tensor;
use std::time::{Instant, Duration};

pub enum TimingsReportItemType {
    ForwardKV,
    LogitsCalc,
    Sampling,
}

pub struct TimingsReportItem {
    item_type: TimingsReportItemType,
    token_index: usize,
    time_range: (Instant, Instant),
}

pub struct SamplingResult {
    pub output_tokens: Vec<u32>,
    pub timings_report: Vec<TimingsReportItem>,
}

impl Default for SamplingResult {
    fn default() -> Self {
        Self::new()
    }
}

impl SamplingResult {
    fn new() -> Self {
        Self {
            output_tokens: Vec::<u32>::new(),
            timings_report: Vec::<TimingsReportItem>::new(),
        }
    }

    fn begin(&mut self, item_type: TimingsReportItemType, token_index: usize) {
        self.timings_report.push(TimingsReportItem {
            item_type,
            token_index,
            time_range: (Instant::now(), Instant::now()),
        });
    }

    fn end(&mut self) {
        let last_index = self.timings_report.len() - 1;
        self.timings_report[last_index].time_range.1 = Instant::now();
    }

    pub fn total_dur(&self) -> Duration {
        let mut mn = self.timings_report[0].time_range.0;
        let mut mx = self.timings_report[0].time_range.1;
        for i in 1..self.timings_report.len() {
            if mn > self.timings_report[i].time_range.0 {
                mn = self.timings_report[i].time_range.0;
            }
            if mx < self.timings_report[i].time_range.1 {
                mx = self.timings_report[i].time_range.1;
            }
        }
        mx - mn
    }
}

pub fn sampling(
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
    model.init_runners(1)?;
    let encoder_output = model.runners[0].write().unwrap().encode(&input_tokens)?;
    for i in 0..max_tokens {
        result.begin(TimingsReportItemType::ForwardKV, i);
        model.runners[0].write().unwrap().forward_kv_cache(
            i..i + 1,
            &encoder_output,
            &output_tokens,
            model.config.use_cache,
        )?;
        result.end();
        result.begin(TimingsReportItemType::LogitsCalc, i);
        let logits = model.runners[0]
            .write()
            .unwrap()
            .get_logits(output_tokens.as_slice())?;
        result.end();
        result.begin(TimingsReportItemType::Sampling, i);
        let p = model.p_from_logits(&logits)?;
        let next_token = model.sample_from_p(&p)?;
        result.end();
        if next_token as usize == model.config.eos_token_id {
            break;
        }
        output_tokens.push(next_token);
    }
    result.output_tokens = output_tokens;
    Ok(result)
}
