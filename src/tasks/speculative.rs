use crate::t5::runner::T5Runner;
use crate::t5::T5Model;
use anyhow::Result;
use candle_core::Tensor;
use crossbeam_epoch::{pin, Atomic, Owned};
use std::sync::atomic::{AtomicBool, Ordering};
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

struct WorkerControlBlock {
    kill: AtomicBool,
    reset: AtomicBool,
    params: Arc<Atomic<(Option<usize>, bool)>>, // DS(n-1) && TF(n-1)
    encoder_output: Arc<Tensor>,
    output_tokens: Arc<RwLock<Vec<u32>>>,
    runner: Arc<RwLock<T5Runner>>,
    next_params: Arc<Atomic<(Option<usize>, bool)>>,
    next_runner: Option<Arc<RwLock<T5Runner>>>,
    logits: Arc<Atomic<Tensor>>,
    result: Arc<RwLock<SamplingResult>>,
}

fn worker(wcb: Arc<WorkerControlBlock>) -> Result<()> {
    let guard = &pin();
    wcb.kill.store(false, Ordering::Release);
    'l: loop {
        while {
            if wcb.kill.load(Ordering::Acquire) {
                break 'l;
            }
            if wcb.reset.load(Ordering::Acquire) {
                wcb.reset.store(false, Ordering::Release);
                wcb.params
                    .store(Owned::new((None, false)), Ordering::Release);
                true
            } else {
                let params =
                    unsafe { wcb.params.load(Ordering::Acquire, guard).deref() };
                params.0.is_none() || !params.1
            }
        } {}
        let params = unsafe { wcb.params.load(Ordering::Acquire, guard).deref() };
        let index = params.0.unwrap();
        let mut runner_write = wcb.runner.write().unwrap();
        let mut result_write = wcb.result.write().unwrap();
        // ForwardKV
        result_write.begin(RunnerType::Target, ActionType::ForwardKV, index);
        runner_write.forward_kv_cache(
            index - 1,
            &wcb.encoder_output,
            &wcb.output_tokens,
        )?;
        result_write.end();
        if wcb.reset.load(Ordering::Acquire) {
            wcb.reset.store(false, Ordering::Release);
            wcb.params
                .store(Owned::new((None, false)), Ordering::Release);
            continue;
        }
        if let Some(next_runner) = &wcb.next_runner {
            next_runner
                .write()
                .unwrap()
                .import_kv_cache(&runner_write.export_kv_cache())?;
        }
        let mut next_params_shared = wcb.next_params.load(Ordering::Acquire, guard);
        while {
            let next_params = unsafe { next_params_shared.deref() };
            let new_params = Owned::new((next_params.0, true));
            match wcb.next_params.compare_exchange(
                next_params_shared,
                new_params,
                Ordering::AcqRel,
                Ordering::Relaxed,
                guard,
            ) {
                Ok(_) => false,
                Err(p) => {
                    next_params_shared = p.current;
                    drop(p.new);
                    true
                }
            }
        } {}
        if wcb.reset.load(Ordering::Acquire) {
            wcb.reset.store(false, Ordering::Release);
            wcb.params
                .store(Owned::new((None, false)), Ordering::Release);
            continue;
        }
        // LogitsCalc
        result_write.begin(RunnerType::Target, ActionType::LogitsCalc, index);
        let logits = runner_write.get_logits(&wcb.output_tokens)?;
        wcb.logits.store(Owned::new(logits), Ordering::Release);
        result_write.end();
        wcb.reset.store(false, Ordering::Release);
        wcb.params.store(Owned::new((None, false)), Ordering::Release);
    }
    Ok(())
}

pub fn sampling(
    mut draft_model: T5Model,
    mut target_model: T5Model,
    gamma: usize,
    tokens: &[u32],
    max_tokens: usize,
) -> Result<SamplingResult> {
    let guard = &pin();

    let mut result = SamplingResult::new();
    let output_tokens = [target_model
        .config
        .decoder_start_token_id
        .unwrap_or(target_model.config.pad_token_id)
        as u32]
    .to_vec();
    let eos_token_id = target_model.config.eos_token_id as u32;

    let input_tokens = Tensor::new(tokens, &draft_model.device)?.unsqueeze(0)?;
    draft_model.init_runners(2)?;
    let draft_encoder_output = draft_model.runners[0]
        .write()
        .unwrap()
        .encode(&input_tokens)?;
    let draft_encoder_output = Arc::new(draft_encoder_output);
    draft_model.propagate_kv_cache(0)?;

    let input_tokens = Tensor::new(tokens, &target_model.device)?.unsqueeze(0)?;
    target_model.init_runners(gamma + 1)?;
    let target_encoder_output = target_model.runners[0]
        .write()
        .unwrap()
        .encode(&input_tokens)?;
    let target_encoder_output = Arc::new(target_encoder_output);
    target_model.propagate_kv_cache(0)?;

    let mut workers = Vec::new();
    let mut wcbs = Vec::new();
    let mut paramss = Vec::new();
    for _ in 0..gamma + 2 {
        paramss.push(Arc::new(Atomic::new((Option::<usize>::None, false))));
    }
    let output_tokens = Arc::new(RwLock::new(output_tokens));
    for i in 0..gamma + 1 {
        wcbs.push(Arc::new(WorkerControlBlock {
            kill: AtomicBool::new(true),
            reset: AtomicBool::new(false),
            params: paramss[i].clone(),
            encoder_output: target_encoder_output.clone(),
            output_tokens: output_tokens.clone(),
            runner: target_model.runners[i].clone(),
            next_params: paramss[i + 1].clone(),
            next_runner: if i < gamma {
                Some(target_model.runners[i + 1].clone())
            } else {
                None
            },
            logits: Arc::new(Atomic::null()),
            result: Arc::new(RwLock::new(SamplingResult::new())),
        }));
        let wcb = wcbs[i].clone();
        workers.push(thread::spawn(move || {
            if let Err(e) = worker(wcb) {
                panic!("worker panic: {:?}", e);
            }
        }));
    }
    for wcb in &wcbs {
        while wcb.kill.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }
    }
    log::info!("Workers ready.");

    while output_tokens.read().unwrap().len() < max_tokens {
        let i = output_tokens.read().unwrap().len();
        let mut qs = Vec::<Vec<f32>>::new();
        let mut new_tokens = Vec::<u32>::new();
        draft_model.pass_kv_cache(0, 1)?;
        wcbs[0]
            .params
            .store(Owned::new((Some(i), true)), Ordering::Release);
        for j in 0..gamma {
            // ForwardKV
            result.begin(RunnerType::Draft, ActionType::ForwardKV, i + j);
            draft_model.runners[1].write().unwrap().forward_kv_cache(
                i + j - 1,
                &draft_encoder_output,
                &output_tokens,
            )?;
            result.end();
            // LogitsCalc
            result.begin(RunnerType::Draft, ActionType::LogitsCalc, i + j);
            let logits = draft_model.runners[1]
                .write()
                .unwrap()
                .get_logits(&output_tokens)?;
            result.end();
            // Sampling
            result.begin(RunnerType::Draft, ActionType::Sampling, i + j);
            qs.push(draft_model.p_from_logits(&logits)?);
            let next_token = draft_model.sample_from_p(&qs[j])?;
            result.end();
            output_tokens.write().unwrap().push(next_token);
            new_tokens.push(next_token);
            if next_token == eos_token_id {
                break;
            }
        }
        let cur_gamma = new_tokens.len();
        for j in 0..cur_gamma {
            // Launch target
            let mut params_shared = wcbs[j + 1].params.load(Ordering::Acquire, guard);
            while {
                let params = unsafe { params_shared.deref() };
                let new_params = Owned::new((Some(i + j + 1), params.1));
                match wcbs[j + 1].params.compare_exchange(
                    params_shared,
                    new_params,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                    guard,
                ) {
                    Ok(_) => false,
                    Err(p) => {
                        params_shared = p.current;
                        drop(p.new);
                        true
                    }
                }
            } {}
        }
        let mut accept_cnt = 0;
        for j in 0..cur_gamma {
            while {
                let params_shared = wcbs[j].params.load(Ordering::Acquire, guard);
                let params = unsafe { params_shared.deref() };
                params.0.is_some() || params.1
            } {}
            let logits_shared = wcbs[j].logits.load(Ordering::Acquire, guard);
            let logits = unsafe { logits_shared.deref() };
            let p = target_model.p_from_logits(logits)?;
            let accept_prob = f32::min(
                1_f32,
                p[new_tokens[j] as usize] / qs[j][new_tokens[j] as usize],
            );
            if target_model.prob_test(accept_prob) {
                accept_cnt += 1;
            } else {
                for wcb in wcbs.iter().take(cur_gamma).skip(j + 1) {
                    wcb.reset.store(true, Ordering::Release);
                }
                break;
            }
        }
        if output_tokens.read().unwrap()[i + accept_cnt - 1] == eos_token_id {
            break;
        }
        let new_token = if accept_cnt == cur_gamma {
            while {
                let params_shared =
                    wcbs[cur_gamma].params.load(Ordering::Acquire, guard);
                let params = unsafe { params_shared.deref() };
                params.0.is_some() || params.1
            } {}
            let logits_shared = wcbs[cur_gamma].logits.load(Ordering::Acquire, guard);
            let logits = unsafe { logits_shared.deref() };
            // Upsample
            result.begin(
                RunnerType::Target,
                ActionType::Sampling,
                i + accept_cnt,
            );
            let p = target_model.p_from_logits(logits)?;
            let new_token = target_model.sample_from_p(&p)?;
            result.end();
            new_token
        } else {
            let logits_shared =
                wcbs[accept_cnt].logits.load(Ordering::Acquire, guard);
            let logits = unsafe { logits_shared.deref() };
            // Resample
            result.begin(
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
            result.end();
            new_token
        };
        if new_token == eos_token_id {
            break;
        }
        for wcb in &wcbs {
            let params_shared = wcb.params.load(Ordering::Acquire, guard);
            let params = unsafe { params_shared.deref() };
            if params.0.is_some() || params.1 {
                wcb.reset.store(true, Ordering::Release);
            }
        }
        for wcb in &wcbs {
            while {
                let params_shared = wcb.params.load(Ordering::Acquire, guard);
                let params = unsafe { params_shared.deref() };
                params.0.is_some() || params.1
            } {
                if !wcb.reset.load(Ordering::Acquire) {
                    wcb.params.store(Owned::new((None, false)), Ordering::Release);
                }
                std::hint::spin_loop();
            }
        }
        log::info!("accept_cnt : {:?}", accept_cnt);
        target_model.pass_kv_cache(accept_cnt, 0)?;
        let mut output_tokens_write = output_tokens.write().unwrap();
        output_tokens_write.truncate(i + accept_cnt);
        output_tokens_write.push(new_token);
        drop(output_tokens_write);
        if draft_model.config.use_cache {
            // ForwardKV
            if accept_cnt == cur_gamma {
                result.begin(
                    RunnerType::Draft,
                    ActionType::ForwardKV,
                    output_tokens.read().unwrap().len() - 2,
                );
                draft_model.runners[0].write().unwrap().forward_kv_cache(
                    output_tokens.read().unwrap().len() - 2,
                    &draft_encoder_output,
                    &output_tokens,
                )?;
                result.end();
            } else {
                for j in i - 1..output_tokens.read().unwrap().len() - 1 {
                    result.begin(RunnerType::Draft, ActionType::ForwardKV, j);
                    draft_model.runners[0].write().unwrap().forward_kv_cache(
                        j,
                        &draft_encoder_output,
                        &output_tokens,
                    )?;
                    result.end();
                }
            };
        }
    }
    for wcb in &wcbs {
        wcb.kill.store(true, Ordering::Release);
    }
    while let Some(w) = workers.pop() {
        let _ = w.join();
    }
    result.output_tokens.clone_from(&output_tokens.read().unwrap());
    for wcb in &wcbs {
        for item in &wcb.result.read().unwrap().timings_report {
            result.timings_report.push(item.clone());
        }
    }
    result.timings_report.sort_by(|a, b| a.time_range.0.cmp(&b.time_range.0));
    Ok(result)
}
