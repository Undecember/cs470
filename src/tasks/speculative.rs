use crate::t5::T5Model;
use anyhow::{Error as E, Result};
use candle_core::Tensor;
use std::sync::{Arc, RwLock};
use std::thread::scope;
use std::time::{Instant, Duration};

#[derive(Debug)]
pub enum RunnerType {
    Draft,
    Target,
}

#[derive(Debug)]
pub enum ActionType {
    ForwardKV,
    LogitsCalc,
    Sampling,
}

#[derive(Debug)]
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
}

pub fn sampling(
    draft_model: &mut T5Model,
    target_model: &mut T5Model,
    gamma: usize,
    tokens: &[u32],
    max_tokens: usize,
) -> Result<SamplingResult> {
    let mut target_model = Arc::new(target_model);

    let mut result = SamplingResult::new();
    let mut output_tokens = [target_model
        .config
        .decoder_start_token_id
        .unwrap_or(target_model.config.pad_token_id)
        as u32]
    .to_vec();

    let input_tokens = Tensor::new(tokens, &draft_model.device)?.unsqueeze(0)?;
    draft_model.init_runners(2)?;
    let draft_encoder_output = draft_model.runners[0]
        .write()
        .unwrap()
        .encode(&input_tokens)?;
    draft_model.propagate_kv_cache(0)?;

    let input_tokens = Tensor::new(tokens, &target_model.device)?.unsqueeze(0)?;
    let target_mut = Arc::get_mut(&mut target_model).unwrap();
    target_mut.init_runners(gamma + 1)?;
    let target_encoder_output = target_mut.runners[0]
        .write()
        .unwrap()
        .encode(&input_tokens)?;
    target_mut.propagate_kv_cache(0)?;

    while output_tokens.len() < max_tokens {
        let i = output_tokens.len();
        let mut qs = Vec::<Vec<f32>>::new();
        let mut new_tokens = Vec::<u32>::new();
        draft_model.pass_kv_cache(0, 1)?;
        for j in 0..gamma {
            result.begin(RunnerType::Draft, ActionType::ForwardKV, i + j);
            draft_model.runners[1].write().unwrap().forward_kv_cache(
                i + j - 1..i + j,
                &draft_encoder_output,
                &output_tokens,
                draft_model.config.use_cache,
            )?;
            result.end();
            result.begin(RunnerType::Draft, ActionType::LogitsCalc, i + j);
            let logits = draft_model.runners[1]
                .write()
                .unwrap()
                .get_logits(output_tokens.as_slice())?;
            result.end();
            result.begin(RunnerType::Draft, ActionType::Sampling, i + j);
            qs.push(draft_model.p_from_logits(&logits)?);
            let next_token = draft_model.sample_from_p(&qs[j])?;
            result.end();
            output_tokens.push(next_token);
            new_tokens.push(next_token);
            if next_token as usize == draft_model.config.eos_token_id {
                break;
            }
        }
        let cur_gamma = new_tokens.len();
        let mut ps = Vec::<RwLock<Result<Vec<f32>>>>::new();
        let mut timings = Vec::<RwLock<SamplingResult>>::new();
        for _ in 0..cur_gamma + 1 {
            ps.push(RwLock::new(Ok(Vec::<f32>::new())));
            timings.push(RwLock::new(SamplingResult::new()));
        }
        let ps = Arc::new(ps);
        let timings = Arc::new(timings);
        scope(|s| {
            for j in 0..cur_gamma + 1 {
                result.begin(RunnerType::Target, ActionType::ForwardKV, i + j);
                let _ = target_model.runners[j].write().unwrap().forward_kv_cache(
                    i + j - 1..i + j,
                    &target_encoder_output,
                    &output_tokens,
                    target_model.config.use_cache,
                );
                result.end();
                if j < cur_gamma {
                    let _ = target_model.pass_kv_cache(j, j + 1);
                }

                let ps = ps.clone();
                let target_model = target_model.clone();
                let timings = timings.clone();
                let output_slice = output_tokens.as_slice();
                s.spawn(move || {
                    let mut timings = timings[j].write().unwrap();
                    timings.begin(RunnerType::Target, ActionType::LogitsCalc, i + j);
                    let logits = target_model.runners[j]
                        .write()
                        .unwrap()
                        .get_logits(output_slice);
                    if let Err(e) = logits {
                        *ps[j].write().unwrap() = Err(E::msg(e));
                        return;
                    }
                    let logits = logits.unwrap();
                    timings.end();
                    let p = target_model.p_from_logits(&logits);
                    *ps[j].write().unwrap() = p;
                });
            }
        });
        let rps = Arc::into_inner(ps).unwrap();
        let mut ps = Vec::<Vec<f32>>::new();
        for r in rps {
            ps.push(r.into_inner().unwrap()?);
        }
        let timings = Arc::into_inner(timings).unwrap();
        for t in timings {
            for item in t.into_inner().unwrap().timings_report {
                result.timings_report.push(item);
            }
        }
        output_tokens.truncate(output_tokens.len() - cur_gamma);
        let target_mut = Arc::get_mut(&mut target_model).unwrap();
        let mut accept_cnt = 0;
        for j in 0..cur_gamma {
            let accept_prob = f32::min(
                1_f32,
                ps[j][new_tokens[j] as usize] / qs[j][new_tokens[j] as usize],
            );
            if target_mut.prob_test(accept_prob) {
                output_tokens.push(new_tokens[j]);
                accept_cnt += 1;
            } else {
                break;
            }
        }
        if let Some(token) = output_tokens.last() {
            if *token as usize == target_mut.config.eos_token_id {
                break;
            }
        }
        if accept_cnt == cur_gamma {
            result.begin(
                RunnerType::Target,
                ActionType::Sampling,
                output_tokens.len(),
            );
            let new_token = target_mut.sample_from_p(&ps[cur_gamma])?;
            result.end();
            output_tokens.push(new_token);
        } else {
            result.begin(
                RunnerType::Target,
                ActionType::Sampling,
                output_tokens.len(),
            );
            let p: Vec<f32> = ps[accept_cnt]
                .iter()
                .zip(&qs[accept_cnt])
                .map(|(p, q)| f32::max(0 as f32, *p - *q))
                .collect();
            let new_token = target_mut.sample_from_p(&p)?;
            result.end();
            output_tokens.push(new_token);
        }
        if let Some(token) = output_tokens.last() {
            if *token as usize == target_mut.config.eos_token_id {
                break;
            }
        }
        target_mut.pass_kv_cache(accept_cnt, 0)?;
        if draft_model.config.use_cache {
            if accept_cnt == cur_gamma {
                draft_model.runners[0].write().unwrap().forward_kv_cache(
                    output_tokens.len() - 2..output_tokens.len() - 1,
                    &draft_encoder_output,
                    &output_tokens,
                    draft_model.config.use_cache,
                )?;
            } else {
                draft_model.runners[0].write().unwrap().forward_kv_cache(
                    i - 1..output_tokens.len() - 1,
                    &draft_encoder_output,
                    &output_tokens,
                    draft_model.config.use_cache,
                )?;
            };
        }
    }
    result.output_tokens = output_tokens;
    Ok(result)
}
