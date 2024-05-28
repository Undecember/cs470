use crate::t5::T5Model;
use anyhow::Result;
use candle_core::Tensor;
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub enum RunnerType {
    Draft,
    Target,
}

#[derive(Clone, Debug)]
pub enum ActionType {
    ForwardKV,
    LogitsCalc,
    Sampling,
}

#[derive(Clone, Debug)]
pub struct TimingsReportItem {
    runner_type: RunnerType,
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

    fn begin(
        &mut self,
        runner_type: RunnerType,
        item_type: ActionType,
        token_index: usize,
    ) {
        self.timings_report.push(TimingsReportItem {
            runner_type,
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

    pub fn export_timings(&self, file_path: &str) -> Result<()> {
        let mut buf = String::new();

        let mut start_time = self.timings_report[0].time_range.0;
        for item in &self.timings_report {
            if start_time > item.time_range.0 {
                start_time = item.time_range.0;
            }
        }
        for item in &self.timings_report {
            buf += format!("{} ", item.token_index).as_str();
            buf += match item.runner_type {
                RunnerType::Draft => "draft",
                RunnerType::Target => "target",
            };
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
    mut draft_model: T5Model,
    mut target_model: T5Model,
    gamma: usize,
    tokens: &[u32],
    max_tokens: usize,
) -> Result<SamplingResult> {
    let result = Arc::new(RwLock::new(SamplingResult::new()));
    let output_tokens = [target_model
        .config
        .decoder_start_token_id
        .unwrap_or(target_model.config.pad_token_id) as u32]
    .to_vec();
    let eos_token_id = target_model.config.eos_token_id as u32;

    let input_tokens = Tensor::new(tokens, &draft_model.device)?.unsqueeze(0)?;
    let mut draft_runner_write = draft_model.runner.write().unwrap();
    draft_runner_write.clear_kv_cache();
    let draft_encoder_output = draft_runner_write.encode(&input_tokens)?;
    drop(draft_runner_write);
    let draft_encoder_output = Arc::new(draft_encoder_output);

    let input_tokens = Tensor::new(tokens, &target_model.device)?.unsqueeze(0)?;
    let mut target_runner_write = target_model.runner.write().unwrap();
    target_runner_write.clear_kv_cache();
    let target_encoder_output = target_runner_write.encode(&input_tokens)?;
    drop(target_runner_write);
    let target_encoder_output = Arc::new(target_encoder_output);

    let output_tokens = Arc::new(RwLock::new(output_tokens));

    while output_tokens.read().unwrap().len() < max_tokens {
        let i = output_tokens.read().unwrap().len();
        // Draft
        let mut result_write = result.write().unwrap();
        let mut qs = Vec::new();
        let mut new_tokens = Vec::new();
        for j in 0..gamma {
            // ForwardKV
            result_write.begin(RunnerType::Draft, ActionType::ForwardKV, i + j);
            let draft_decoder_output =
                draft_model.runner.write().unwrap().forward_kv_cache(
                    i + j - 1..i + j,
                    &draft_encoder_output,
                    &output_tokens,
                )?;
            result_write.end();
            // LogitsCalc
            result_write.begin(RunnerType::Draft, ActionType::LogitsCalc, i + j);
            let logits = draft_model
                .runner
                .write()
                .unwrap()
                .get_logits(draft_decoder_output, &output_tokens)?;
            result_write.end();
            // Sampling
            result_write.begin(RunnerType::Draft, ActionType::Sampling, i + j);
            qs.push(draft_model.p_from_logits(&logits)?);
            let next_token = draft_model.sample_from_p(&qs[j])?;
            result_write.end();
            output_tokens.write().unwrap().push(next_token);
            new_tokens.push(next_token);
            if next_token == eos_token_id {
                break;
            }
        }
        let cur_gamma = new_tokens.len();
        // Target (KV)
        let mut target_runner_write = target_model.runner.write().unwrap();
        result_write.begin(RunnerType::Target, ActionType::ForwardKV, i + cur_gamma);
        let target_decoder_outputs = target_runner_write.forward_kv_cache(
            i - 1..i + cur_gamma,
            &target_encoder_output,
            &output_tokens,
        )?;
        let mut target_decoder_outputs = {
            let mut res = Vec::<Tensor>::with_capacity(cur_gamma + 1);
            for j in 0..cur_gamma + 1 {
                res.push(
                    target_decoder_outputs
                        .get_on_dim(1, j)?
                        .unsqueeze(0)?,
                );
            }
            res
        };
        result_write.end();
        drop(target_runner_write);
        drop(result_write);
        // Target (parallel)
        let target_logitss = Arc::new(RwLock::new(Vec::new()));
        thread::scope(|s| {
            for j in 0..cur_gamma + 1 {
                let logitss = target_logitss.clone();
                let result = result.clone();
                let runner = target_model.runner.clone();
                let output_tokens = output_tokens.clone();
                let decoder_output = if let Some(d) = target_decoder_outputs.pop() {
                    d
                } else {
                    panic!("Missing target decoder output index {}, {}", i, j);
                };
                s.spawn(move || {
                    // LogitsCalc
                    let mut report = TimingsReportItem {
                        runner_type: RunnerType::Target,
                        item_type: ActionType::LogitsCalc,
                        token_index: i + j,
                        time_range: (Instant::now(), Instant::now()),
                    };
                    let logits = if let Ok(l) = runner
                        .read()
                        .unwrap()
                        .get_logits(decoder_output, &output_tokens)
                    {
                        l
                    } else {
                        panic!("Failed to get logit index {}, {}", i, j);
                    };
                    report.time_range.1 = Instant::now();
                    result.write().unwrap().timings_report.push(report);
                    logitss.write().unwrap().push((j, logits));
                });
            }
        });
        let target_logits = {
            let mut res = Vec::new();
            target_logitss
                .write()
                .unwrap()
                .sort_by(|a, b| a.0.cmp(&b.0));
            while let Some((_, logits)) = target_logitss.write().unwrap().pop() {
                res.push(logits);
            }
            res
        };
        let mut accept_cnt = 0;
        for j in 0..cur_gamma {
            let logits = &target_logits[j];
            let p = target_model.p_from_logits(logits)?;
            let accept_prob = f32::min(
                1_f32,
                p[new_tokens[j] as usize] / qs[j][new_tokens[j] as usize],
            );
            if target_model.prob_test(accept_prob) {
                accept_cnt += 1;
            } else {
                break;
            }
        }
        if output_tokens.read().unwrap()[i + accept_cnt - 1] == eos_token_id {
            break;
        }
        let mut result_write = result.write().unwrap();
        let new_token = if accept_cnt == cur_gamma {
            let logits = &target_logits[cur_gamma];
            // Upsample
            result_write.begin(
                RunnerType::Target,
                ActionType::Sampling,
                i + accept_cnt,
            );
            let p = target_model.p_from_logits(logits)?;
            let new_token = target_model.sample_from_p(&p)?;
            result_write.end();
            new_token
        } else {
            let logits = &target_logits[accept_cnt];
            // Resample
            result_write.begin(
                RunnerType::Target,
                ActionType::Sampling,
                i + accept_cnt,
            );
            let p = target_model.p_from_logits(logits)?;
            let p: Vec<f32> = p
                .iter()
                .zip(&qs[accept_cnt])
                .map(|(p, q)| f32::max(0 as f32, *p - *q))
                .collect();
            let new_token = target_model.sample_from_p(&p)?;
            result_write.end();
            new_token
        };
        if new_token == eos_token_id {
            break;
        }
        let mut output_tokens_write = output_tokens.write().unwrap();
        output_tokens_write.truncate(i + accept_cnt);
        output_tokens_write.push(new_token);
        drop(output_tokens_write);
        target_model
            .runner
            .write()
            .unwrap()
            .rollback_kv_cache(cur_gamma - accept_cnt)?;
        if accept_cnt == cur_gamma {
            // ForwardKV
            result_write.begin(
                RunnerType::Draft,
                ActionType::ForwardKV,
                i + accept_cnt,
            );
            draft_model.runner.write().unwrap().forward_kv_cache(
                i + accept_cnt - 1..i + accept_cnt,
                &draft_encoder_output,
                &output_tokens,
            )?;
            result_write.end();
        }
        if accept_cnt + 1 < cur_gamma {
            draft_model
                .runner
                .write()
                .unwrap()
                .rollback_kv_cache(cur_gamma - accept_cnt - 1)?;
        }
        drop(result_write);
    }
    let mut result_write = result.write().unwrap();
    result_write
        .output_tokens
        .clone_from(&output_tokens.read().unwrap());
    result_write
        .timings_report
        .sort_by(|a, b| a.time_range.0.cmp(&b.time_range.0));
    drop(result_write);
    let result = Arc::into_inner(result).unwrap();
    let result = RwLock::into_inner(result).unwrap();
    Ok(result)
}
