use super::report::{ActionType, RunnerType, TaskReport};
use crate::cmd_args::Args;
use crate::hf_models::t5::T5Model;
use anyhow::Result;
use candle_core::Tensor;

pub fn sampling(
    draft_model: &mut T5Model,
    target_model: &mut T5Model,
    args: &Args,
    tokens: &[u32],
    max_tokens: usize,
    kl_epsilon: Option<f64>,
) -> Result<TaskReport> {
    let mut gamma = args.gamma;
    let mut gamma_f = gamma as f64;
    let mut report = TaskReport::new();
    report.output_tokens.push(
        target_model
            .config
            .decoder_start_token_id
            .unwrap_or(target_model.config.pad_token_id) as u32,
    );
    report.accept_report = Some(Vec::new());
    if kl_epsilon.is_some() {
        report.kl_divs = Some((Vec::new(), Vec::new()));
    }
    let eos_token_id = target_model.config.eos_token_id as u32;

    let input_tokens = Tensor::new(tokens, &draft_model.device)?.unsqueeze(0)?;
    draft_model.runner.clear_kv_cache();
    draft_model.reset_rng();
    let draft_encoder_output = draft_model.runner.encode(&input_tokens)?;

    let input_tokens = Tensor::new(tokens, &target_model.device)?.unsqueeze(0)?;
    target_model.runner.clear_kv_cache();
    target_model.reset_rng();
    let target_encoder_output = target_model.runner.encode(&input_tokens)?;

    while report.output_tokens.len() < max_tokens {
        let i = report.output_tokens.len();
        // Draft
        let mut qs = Vec::new();
        let mut new_tokens = Vec::new();
        let mut early_reject_index = gamma;
        for j in 0..gamma {
            // ForwardKV
            report.start(
                RunnerType::Draft,
                ActionType::ForwardKV,
                (i + j, i + j + 1),
            );
            let draft_decoder_output = draft_model.runner.forward_kv_cache(
                i + j - 1..i + j,
                &draft_encoder_output,
                report.output_tokens.as_slice(),
            )?;
            report.end();
            // LogitsCalc
            report.start(
                RunnerType::Draft,
                ActionType::LogitsCalc,
                (i + j, i + j + 1),
            );
            let logits = draft_model.runner.get_logits(draft_decoder_output)?;
            report.end();
            // Sampling
            report.start(RunnerType::Draft, ActionType::Sampling, (i + j, i + j + 1));
            qs.push(draft_model.p_from_logits(
                &logits,
                0,
                report.output_tokens.as_slice(),
            )?);
            let next_token = draft_model.sample_from_p(&qs[j])?;
            report.end();
            report.output_tokens.push(next_token);
            new_tokens.push(next_token);
            if qs[j][next_token as usize] < args.early_reject_thr as f32 {
                early_reject_index = j;
                break;
            }
            if next_token == eos_token_id {
                break;
            }
        }
        let cur_gamma = new_tokens.len();
        // Target (KV)
        report.start(
            RunnerType::Target,
            ActionType::ForwardKV,
            (i, i + cur_gamma + 1),
        );
        let target_decoder_outputs = target_model.runner.forward_kv_cache(
            i - 1..i + cur_gamma,
            &target_encoder_output,
            report.output_tokens.as_slice(),
        )?;
        report.end();
        // Target (LogitsCalc)
        report.start(
            RunnerType::Target,
            ActionType::LogitsCalc,
            (i, i + cur_gamma + 1),
        );
        let target_logitss =
            target_model.runner.get_logits(target_decoder_outputs)?;
        report.end();
        let mut accept_cnt: usize = 0;
        for j in 0..cur_gamma {
            let p = target_model.p_from_logits(
                &target_logitss,
                j,
                &report.output_tokens[..i + j],
            )?;
            let accept_prob = f32::min(
                1_f32,
                p[new_tokens[j] as usize]
                    / qs[j][new_tokens[j] as usize]
                    / args.lenience as f32,
            );
            let skip = (i + j) % args.sparse_validation > 0;
            if target_model.prob_test(accept_prob) && j != early_reject_index {
                if let Some((kl_divs, _)) = report.kl_divs.as_mut() {
                    kl_divs.push(kl_div(
                        p.as_slice(),
                        qs[j].as_slice(),
                        kl_epsilon.unwrap(),
                    ));
                }
                if !skip || accept_cnt == j {
                    accept_cnt = j + 1;
                }
            } else {
                if let Some((_, kl_divs)) = report.kl_divs.as_mut() {
                    kl_divs.push(kl_div(
                        p.as_slice(),
                        qs[j].as_slice(),
                        kl_epsilon.unwrap(),
                    ));
                }
                if !skip {
                    break;
                }
            }
        }
        report
            .accept_report
            .as_mut()
            .unwrap()
            .push((accept_cnt as u32, cur_gamma as u32));
        if report.output_tokens[i + accept_cnt - 1] == eos_token_id {
            break;
        }
        report.start(
            RunnerType::Target,
            ActionType::Sampling,
            (i + accept_cnt, i + accept_cnt + 1),
        );
        let p = target_model.p_from_logits(
            &target_logitss,
            accept_cnt,
            &report.output_tokens[..i + accept_cnt],
        )?;
        let p = if accept_cnt < cur_gamma {
            // Resample
            p.iter()
                .zip(&qs[accept_cnt])
                .map(|(p, q)| f32::max(0 as f32, *p - *q))
                .collect()
        } else {
            // Upsample
            p
        };
        let new_token = target_model.sample_from_p(&p)?;
        report.end();
        report.output_tokens.truncate(i + accept_cnt);
        report.output_tokens.push(new_token);
        if new_token == eos_token_id {
            break;
        }
        target_model
            .runner
            .rollback_kv_cache(cur_gamma - accept_cnt)?;
        if accept_cnt == cur_gamma {
            // ForwardKV
            report.start(
                RunnerType::Draft,
                ActionType::ForwardKV,
                (i + accept_cnt, i + accept_cnt + 1),
            );
            draft_model.runner.forward_kv_cache(
                i + accept_cnt - 1..i + accept_cnt,
                &draft_encoder_output,
                report.output_tokens.as_slice(),
            )?;
            report.end();
        }
        if accept_cnt + 1 < cur_gamma {
            draft_model
                .runner
                .rollback_kv_cache(cur_gamma - accept_cnt - 1)?;
        }
        let alpha = accept_cnt as f64 / (cur_gamma + 1) as f64;
        gamma_f = args.adaptive_gamma_theta * gamma_f
            + (1_f64 - args.adaptive_gamma_theta) * (1_f64 / (1_f64 - alpha));
        gamma = gamma_f.round() as usize;
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
