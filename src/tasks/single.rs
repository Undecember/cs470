use crate::t5::T5Model;
use anyhow::Result;
use candle_core::Tensor;
use std::time::{Duration, Instant};
use std::sync::RwLock;

pub enum ActionType {
    ForwardKV,
    LogitsCalc,
    Sampling,
}

pub struct TimingsReportItem {
    item_type: ActionType,
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

    fn begin(&mut self, item_type: ActionType, token_index: usize) {
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

    pub fn export_timings(&self, file_path: &str, runner_type: &str) -> Result<()> {
        let mut buf = String::new();

        let mut start_time = self.timings_report[0].time_range.0;
        for item in &self.timings_report {
            if start_time > item.time_range.0 {
                start_time = item.time_range.0;
            }
        }
        for item in &self.timings_report {
            buf += format!("{} ", item.token_index).as_str();
            buf += runner_type;
            buf += match item.item_type {
                ActionType::ForwardKV => " forward_kv",
                ActionType::LogitsCalc => " logits_calc",
                ActionType::Sampling => " sampling",
            };
            buf += format!(
                " {} {} {}\n",
                (item.time_range.0 - start_time).as_micros(),
                (item.time_range.1 - start_time).as_micros(),
                (item.time_range.1 - item.time_range.0).as_micros(),
            )
            .as_str();
        }

        std::fs::write(file_path, buf.as_str())?;
        Ok(())
    }
}

pub fn sampling(
    model: &mut T5Model,
    tokens: &[u32],
    max_tokens: usize,
) -> Result<SamplingResult> {
    let mut result = SamplingResult::new();
    let output_tokens = [model
        .config
        .decoder_start_token_id
        .unwrap_or(model.config.pad_token_id) as u32]
    .to_vec();

    let input_tokens = Tensor::new(tokens, &model.device)?.unsqueeze(0)?;
    model.init_runners(1)?;
    let encoder_output = model.runners[0].write().unwrap().encode(&input_tokens)?;
    let output_tokens = RwLock::new(output_tokens);
    for i in 0..max_tokens {
        result.begin(ActionType::ForwardKV, i);
        model.runners[0].write().unwrap().forward_kv_cache(
            i,
            &encoder_output,
            &output_tokens,
        )?;
        result.end();
        result.begin(ActionType::LogitsCalc, i);
        let logits = model.runners[0]
            .write()
            .unwrap()
            .get_logits(&output_tokens)?;
        result.end();
        result.begin(ActionType::Sampling, i);
        let p = model.p_from_logits(&logits)?;
        let next_token = model.sample_from_p(&p)?;
        result.end();
        if next_token as usize == model.config.eos_token_id {
            break;
        }
        output_tokens.write().unwrap().push(next_token);
    }
    result.output_tokens.clone_from(&output_tokens.read().unwrap());
    Ok(result)
}
