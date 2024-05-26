use crate::t5::T5Model;
use anyhow::{Error as E, Result};
use candle_core::Tensor;
use std::sync::{Arc, RwLock};
use std::thread::scope;
use std::time::Instant;

#[derive(Debug)]
pub enum TimingsReportItem {
    DraftBegin(usize),
    DraftEnd(usize),
    TargetBegin(usize),
    TargetEnd(usize),
    Accept(usize),
    Reject(usize),
    Resample(usize),
    Upsample(usize),
}

pub struct SamplingResult {
    pub output_tokens: Vec<u32>,
    pub timings_report: Arc<RwLock<Vec<(Instant, TimingsReportItem)>>>,
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
            timings_report: Arc::new(RwLock::new(
                Vec::<(Instant, TimingsReportItem)>::new(),
            )),
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
    model.init_runners(1)?;
    let encoder_output = model.runners[0].write().unwrap().encode(&input_tokens)?;
    for i in 0..max_tokens {
        let begin_time = Instant::now();
        model.runners[0].write().unwrap().forward_kv_cache(
            i..i + 1,
            &encoder_output,
            &output_tokens,
            model.config.use_cache,
        )?;
        let logits = model.runners[0]
            .write()
            .unwrap()
            .get_logits(output_tokens.as_slice())?;
        let p = model.p_from_logits(&logits)?;
        let next_token = model.sample_from_p(&p)?;
        let end_time = Instant::now();
        let mut timings_report_write = result.timings_report.write().unwrap();
        timings_report_write.push((begin_time, TimingsReportItem::TargetBegin(i)));
        timings_report_write.push((end_time, TimingsReportItem::TargetEnd(i)));
        drop(timings_report_write);
        if next_token as usize == model.config.eos_token_id {
            break;
        }
        output_tokens.push(next_token);
    }
    result.output_tokens = output_tokens;
    Ok(result)
}

pub fn speculative_sampling(
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
            let begin_time = Instant::now();
            draft_model.runners[1].write().unwrap().forward_kv_cache(
                i + j - 1..i + j,
                &draft_encoder_output,
                &output_tokens,
                draft_model.config.use_cache,
            )?;
            let logits = draft_model.runners[1]
                .write()
                .unwrap()
                .get_logits(output_tokens.as_slice())?;
            qs.push(draft_model.p_from_logits(&logits)?);
            let next_token = draft_model.sample_from_p(&qs[j])?;
            let end_time = Instant::now();
            let mut timings_report_write = result.timings_report.write().unwrap();
            timings_report_write
                .push((begin_time, TimingsReportItem::DraftBegin(i + j)));
            timings_report_write.push((end_time, TimingsReportItem::DraftEnd(i + j)));
            drop(timings_report_write);
            output_tokens.push(next_token);
            new_tokens.push(next_token);
            if next_token as usize == draft_model.config.eos_token_id {
                break;
            }
        }
        let cur_gamma = new_tokens.len();
        let mut ps = Vec::<RwLock<Result<Vec<f32>>>>::new();
        for _ in 0..cur_gamma + 1 {
            ps.push(RwLock::new(Ok(Vec::<f32>::new())));
        }
        let ps = Arc::new(ps);
        scope(|s| {
            for j in 0..cur_gamma + 1 {
                let mut timings_report_write = result.timings_report.write().unwrap();
                timings_report_write
                    .push((Instant::now(), TimingsReportItem::TargetBegin(0)));
                let _ = target_model.runners[j].write().unwrap().forward_kv_cache(
                    i + j - 1..i + j,
                    &target_encoder_output,
                    &output_tokens,
                    target_model.config.use_cache,
                );
                if j < cur_gamma {
                    let _ = target_model.pass_kv_cache(j, j + 1);
                }

                let ps = ps.clone();
                let target_model = target_model.clone();
                let timings_report = result.timings_report.clone();
                let output_slice = output_tokens.as_slice();
                s.spawn(move || {
                    let begin_time = Instant::now();
                    let logits = target_model.runners[j]
                        .write()
                        .unwrap()
                        .get_logits(output_slice);
                    if let Err(e) = logits {
                        *ps[j].write().unwrap() = Err(E::msg(e));
                        return;
                    }
                    let logits = logits.unwrap();
                    let p = target_model.p_from_logits(&logits);
                    *ps[j].write().unwrap() = p;
                    let end_time = Instant::now();
                    let mut timings_report_write = timings_report.write().unwrap();
                    timings_report_write
                        .push((begin_time, TimingsReportItem::TargetBegin(i + j)));
                    timings_report_write
                        .push((end_time, TimingsReportItem::TargetEnd(i + j)));
                });
            }
        });
        let rps = Arc::into_inner(ps).unwrap();
        let mut ps = Vec::<Vec<f32>>::new();
        for r in rps {
            ps.push(r.into_inner().unwrap()?);
        }
        output_tokens.truncate(output_tokens.len() - cur_gamma);
        let target_mut = Arc::get_mut(&mut target_model).unwrap();
        let mut accept_cnt = 0;
        for j in 0..cur_gamma {
            let accept_prob = f32::min(
                1_f32,
                ps[j][new_tokens[j] as usize] / qs[j][new_tokens[j] as usize],
            );
            let mut timings_report_write = result.timings_report.write().unwrap();
            if target_mut.prob_test(accept_prob) {
                output_tokens.push(new_tokens[j]);
                accept_cnt += 1;
                timings_report_write
                    .push((Instant::now(), TimingsReportItem::Accept(i + j)));
            } else {
                timings_report_write
                    .push((Instant::now(), TimingsReportItem::Reject(i + j)));
                break;
            }
        }
        if let Some(token) = output_tokens.last() {
            if *token as usize == target_mut.config.eos_token_id {
                break;
            }
        }
        if accept_cnt == cur_gamma {
            let new_token = target_mut.sample_from_p(&ps[cur_gamma])?;
            let mut timings_report_write = result.timings_report.write().unwrap();
            timings_report_write.push((
                Instant::now(),
                TimingsReportItem::Upsample(output_tokens.len()),
            ));
            output_tokens.push(new_token);
        } else {
            let p: Vec<f32> = ps[accept_cnt]
                .iter()
                .zip(&qs[accept_cnt])
                .map(|(p, q)| f32::max(0 as f32, *p - *q))
                .collect();
            let new_token = target_mut.sample_from_p(&p)?;
            let mut timings_report_write = result.timings_report.write().unwrap();
            timings_report_write.push((
                Instant::now(),
                TimingsReportItem::Resample(output_tokens.len()),
            ));
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
