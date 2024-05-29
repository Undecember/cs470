use super::report::{ActionType, RunnerType, TaskReport};
use crate::hf_models::t5::T5Model;
use anyhow::Result;
use candle_core::Tensor;
use std::sync::{Arc, RwLock};

pub fn sampling(
    draft_model: &mut T5Model,
    target_model: &mut T5Model,
    gamma: usize,
    tokens: &[u32],
    max_tokens: usize,
    kl_epsilon: Option<f64>,
) -> Result<TaskReport> {
    let report = Arc::new(TaskReport::new());
    let output_tokens = [target_model
        .config
        .decoder_start_token_id
        .unwrap_or(target_model.config.pad_token_id) as u32]
    .to_vec();
    let mut accept_report = Vec::new();
    let mut kl_divs = (Vec::new(), Vec::new());
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
        let mut qs = Vec::new();
        let mut new_tokens = Vec::new();
        for j in 0..gamma {
            // ForwardKV
            let output_tokens_read = output_tokens.read().unwrap();
            let span = report.start(RunnerType::Draft, ActionType::ForwardKV, i + j);
            let draft_decoder_output =
                draft_model.runner.write().unwrap().forward_kv_cache(
                    i + j - 1..i + j,
                    &draft_encoder_output,
                    output_tokens_read.as_slice(),
                )?;
            report.end(span);
            // LogitsCalc
            let span = report.start(RunnerType::Draft, ActionType::LogitsCalc, i + j);
            let logits = draft_model
                .runner
                .write()
                .unwrap()
                .get_logits(draft_decoder_output)?;
            report.end(span);
            // Sampling
            let span = report.start(RunnerType::Draft, ActionType::Sampling, i + j);
            qs.push(draft_model.p_from_logits(
                &logits,
                0,
                output_tokens_read.as_slice(),
            )?);
            let next_token = draft_model.sample_from_p(&qs[j])?;
            report.end(span);
            drop(output_tokens_read);
            output_tokens.write().unwrap().push(next_token);
            new_tokens.push(next_token);
            if next_token == eos_token_id {
                break;
            }
        }
        let cur_gamma = new_tokens.len();
        // Target (KV)
        let mut target_runner_write = target_model.runner.write().unwrap();
        let span =
            report.start(RunnerType::Target, ActionType::ForwardKV, i + cur_gamma);
        let target_decoder_outputs = target_runner_write.forward_kv_cache(
            i - 1..i + cur_gamma,
            &target_encoder_output,
            output_tokens.read().unwrap().as_slice(),
        )?;
        report.end(span);
        drop(target_runner_write);
        // Target (LogitsCalc)
        let span =
            report.start(RunnerType::Target, ActionType::LogitsCalc, i + cur_gamma);
        let target_logitss = target_model
            .runner
            .read()
            .unwrap()
            .get_logits(target_decoder_outputs)?;
        report.end(span);
        let mut accept_cnt = 0;
        let output_tokens_read = output_tokens.read().unwrap();
        for j in 0..cur_gamma {
            let p = target_model.p_from_logits(
                &target_logitss,
                j,
                output_tokens_read.as_slice(),
            )?;
            let accept_prob = f32::min(
                1_f32,
                p[new_tokens[j] as usize] / qs[j][new_tokens[j] as usize],
            );
            if target_model.prob_test(accept_prob) {
                accept_cnt += 1;
                if let Some(eps) = kl_epsilon {
                    kl_divs.0.push(kl_div(p.as_slice(), qs[j].as_slice(), eps));
                }
            } else {
                if let Some(eps) = kl_epsilon {
                    kl_divs.1.push(kl_div(p.as_slice(), qs[j].as_slice(), eps));
                }
                break;
            }
        }
        accept_report.push((accept_cnt as u32, cur_gamma as u32));
        if output_tokens_read[i + accept_cnt - 1] == eos_token_id {
            break;
        }
        let new_token = if accept_cnt == cur_gamma {
            // Upsample
            let span = report.start(
                RunnerType::Target,
                ActionType::Sampling,
                i + accept_cnt,
            );
            let p = target_model.p_from_logits(
                &target_logitss,
                cur_gamma,
                output_tokens_read.as_slice(),
            )?;
            let new_token = target_model.sample_from_p(&p)?;
            report.end(span);
            new_token
        } else {
            // Resample
            let span = report.start(
                RunnerType::Target,
                ActionType::Sampling,
                i + accept_cnt,
            );
            let p = target_model.p_from_logits(
                &target_logitss,
                accept_cnt,
                output_tokens_read.as_slice(),
            )?;
            let p: Vec<f32> = p
                .iter()
                .zip(&qs[accept_cnt])
                .map(|(p, q)| f32::max(0 as f32, *p - *q))
                .collect();
            let new_token = target_model.sample_from_p(&p)?;
            report.end(span);
            new_token
        };
        drop(output_tokens_read);
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
            let span = report.start(
                RunnerType::Draft,
                ActionType::ForwardKV,
                i + accept_cnt,
            );
            draft_model.runner.write().unwrap().forward_kv_cache(
                i + accept_cnt - 1..i + accept_cnt,
                &draft_encoder_output,
                output_tokens.read().unwrap().as_slice(),
            )?;
            report.end(span);
        }
        if accept_cnt + 1 < cur_gamma {
            draft_model
                .runner
                .write()
                .unwrap()
                .rollback_kv_cache(cur_gamma - accept_cnt - 1)?;
        }
    }
    let mut report = Arc::into_inner(report).unwrap();
    report.set_output_tokens(output_tokens.read().unwrap().as_slice());
    report.set_accept_report(accept_report.as_slice());
    if kl_epsilon.is_some() {
        report.set_kl_divs(kl_divs);
    }
    report.sort_timings();
    Ok(report)
}

fn kl_div(p: &[f32], q: &[f32], eps: f64) -> f64 {
    let p = Vec::from(p);
    let q = Vec::from(q);
    let eps = eps as f32;
    let p: Vec<f32> = p.iter().map(|v| v + eps).collect();
    let q: Vec<f32> = q.iter().map(|v| v + eps).collect();
    let p_sum = p.iter().sum::<f32>();
    let q_sum = p.iter().sum::<f32>();
    let p: Vec<f32> = p.iter().map(|v| v / p_sum).collect();
    let q: Vec<f32> = q.iter().map(|v| v / q_sum).collect();
    p.iter().zip(q).map(|(p, q)| p * (p / q).ln()).sum::<f32>() as f64
}
