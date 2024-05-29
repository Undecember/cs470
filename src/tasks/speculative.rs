use super::report::{ActionType, RunnerType, TaskReport};
use crate::hf_models::t5::T5Model;
use anyhow::Result;
use candle_core::Tensor;
use std::sync::{Arc, RwLock};
use std::thread;

pub fn sampling(
    draft_model: &mut T5Model,
    target_model: &mut T5Model,
    gamma: usize,
    tokens: &[u32],
    max_tokens: usize,
) -> Result<TaskReport> {
    let report = Arc::new(TaskReport::new());
    let output_tokens = [target_model
        .config
        .decoder_start_token_id
        .unwrap_or(target_model.config.pad_token_id) as u32]
    .to_vec();
    let mut accept_report = Vec::new();
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
                .get_logits(draft_decoder_output, output_tokens_read.as_slice())?;
            report.end(span);
            drop(output_tokens_read);
            // Sampling
            let span = report.start(RunnerType::Draft, ActionType::Sampling, i + j);
            qs.push(draft_model.p_from_logits(&logits)?);
            let next_token = draft_model.sample_from_p(&qs[j])?;
            report.end(span);
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
        let mut target_decoder_outputs = {
            let mut res = Vec::<Tensor>::with_capacity(cur_gamma + 1);
            for j in 0..cur_gamma + 1 {
                res.push(target_decoder_outputs.get_on_dim(1, j)?.unsqueeze(0)?);
            }
            res
        };
        report.end(span);
        drop(target_runner_write);
        // Target (parallel)
        let target_logitss = Arc::new(RwLock::new(Vec::new()));
        thread::scope(|s| {
            for j in 0..cur_gamma + 1 {
                let logitss = target_logitss.clone();
                let report = report.clone();
                let runner = target_model.runner.clone();
                let output_tokens = output_tokens.clone();
                let decoder_output = if let Some(d) = target_decoder_outputs.pop() {
                    d
                } else {
                    panic!("Missing target decoder output index {}, {}", i, j);
                };
                s.spawn(move || {
                    // LogitsCalc
                    let span = report.start(
                        RunnerType::Target,
                        ActionType::LogitsCalc,
                        i + j,
                    );
                    let logits = if let Ok(l) = runner.read().unwrap().get_logits(
                        decoder_output,
                        output_tokens.read().unwrap().as_slice(),
                    ) {
                        l
                    } else {
                        panic!("Failed to get logit index {}, {}", i, j);
                    };
                    report.end(span);
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
        accept_report.push((accept_cnt as u32, cur_gamma as u32));
        if output_tokens.read().unwrap()[i + accept_cnt - 1] == eos_token_id {
            break;
        }
        let new_token = if accept_cnt == cur_gamma {
            let logits = &target_logits[cur_gamma];
            // Upsample
            let span = report.start(
                RunnerType::Target,
                ActionType::Sampling,
                i + accept_cnt,
            );
            let p = target_model.p_from_logits(logits)?;
            let new_token = target_model.sample_from_p(&p)?;
            report.end(span);
            new_token
        } else {
            let logits = &target_logits[accept_cnt];
            // Resample
            let span = report.start(
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
            report.end(span);
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
    report.sort_timings();
    Ok(report)
}
