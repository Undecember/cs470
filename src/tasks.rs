use crate::t5_model::T5Model;
use anyhow::Result;
use candle_core::Tensor;
use std::time::Instant;

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
        let next_token = model.sample_from_p(&p)?;
        let end_time = Instant::now();
        result
            .timings_report
            .push((begin_time, TimingsReportItem::TargetBegin(i)));
        result
            .timings_report
            .push((end_time, TimingsReportItem::TargetEnd(i)));
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
    let mut result = SamplingResult::new();
    let mut output_tokens = [target_model
        .config
        .decoder_start_token_id
        .unwrap_or(target_model.config.pad_token_id)
        as u32]
    .to_vec();

    let input_tokens = Tensor::new(tokens, &draft_model.device)?.unsqueeze(0)?;
    let draft_encoder_output = draft_model.cgs[0].encode(&input_tokens)?;
    let target_encoder_output = target_model.cgs[0].encode(&input_tokens)?;
    while output_tokens.len() < max_tokens {
        let i = output_tokens.len() - 1;
        let mut ps = Vec::<Vec<f32>>::new();
        let mut qs = Vec::<Vec<f32>>::new();
        let mut new_tokens = Vec::<u32>::new();
        for j in 0..gamma {
            let decoder_tokens = if i == 0 || !draft_model.config.use_cache {
                Tensor::new(output_tokens.as_slice(), &draft_model.device)?
                    .unsqueeze(0)?
            } else {
                let last_token = *output_tokens.last().unwrap();
                Tensor::new(&[last_token], &draft_model.device)?.unsqueeze(0)?
            };
            let begin_time = Instant::now();
            let logits = draft_model.get_logits(
                0,
                &decoder_tokens,
                &draft_encoder_output,
                &output_tokens,
            )?;
            qs.push(draft_model.p_from_logits(&logits)?);
            let next_token = draft_model.sample_from_p(&qs[j])?;
            let end_time = Instant::now();
            result
                .timings_report
                .push((begin_time, TimingsReportItem::DraftBegin(i + j)));
            result
                .timings_report
                .push((end_time, TimingsReportItem::DraftEnd(i + j)));
            output_tokens.push(next_token);
            new_tokens.push(next_token);
            if next_token as usize == draft_model.config.eos_token_id {
                break;
            }
        }
        let cur_gamma = new_tokens.len();
        output_tokens.truncate(output_tokens.len() - cur_gamma);
        let mut accept_cnt = 0;
        for j in 0..cur_gamma + 1 {
            let decoder_tokens = if i == 0 || !target_model.config.use_cache {
                Tensor::new(output_tokens.as_slice(), &target_model.device)?
                    .unsqueeze(0)?
            } else {
                let last_token = *output_tokens.last().unwrap();
                Tensor::new(&[last_token], &target_model.device)?.unsqueeze(0)?
            };
            let begin_time = Instant::now();
            let logits = target_model.get_logits(
                0,
                &decoder_tokens,
                &target_encoder_output,
                &output_tokens,
            )?;
            ps.push(target_model.p_from_logits(&logits)?);
            let end_time = Instant::now();
            result
                .timings_report
                .push((begin_time, TimingsReportItem::TargetBegin(i + j)));
            result
                .timings_report
                .push((end_time, TimingsReportItem::TargetEnd(i + j)));
            if j < cur_gamma {
                output_tokens.push(new_tokens[j]);
            }
        }
        output_tokens.truncate(output_tokens.len() - cur_gamma);
        for j in 0..cur_gamma {
            let accept_prob = f32::min(
                1 as f32,
                ps[j][new_tokens[j] as usize] / qs[j][new_tokens[j] as usize],
            );
            if target_model.prob_test(accept_prob) {
                output_tokens.push(new_tokens[j]);
                accept_cnt += 1;
                result
                    .timings_report
                    .push((Instant::now(), TimingsReportItem::Accept(i + j)));
            } else {
                result
                    .timings_report
                    .push((Instant::now(), TimingsReportItem::Reject(i + j)));
                break;
            }
        }
        if let Some(token) = output_tokens.last() {
            if *token as usize == target_model.config.eos_token_id {
                break;
            }
        }
        use log::debug;
        debug!("accept_cnt:{:?}", accept_cnt);
        debug!("cur_gamma:{:?}", cur_gamma);
        debug!("ps len:{:?}", ps.len());
        debug!("qs len:{:?}", qs.len());
        if accept_cnt == cur_gamma {
            let new_token = target_model.sample_from_p(&ps[cur_gamma])?;
            result.timings_report.push((
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
            let new_token = target_model.sample_from_p(&p)?;
            result.timings_report.push((
                Instant::now(),
                TimingsReportItem::Resample(output_tokens.len()),
            ));
            output_tokens.push(new_token);
        }
        if let Some(token) = output_tokens.last() {
            if *token as usize == target_model.config.eos_token_id {
                break;
            }
        }
    }
    result.output_tokens = output_tokens;
    Ok(result)
}
