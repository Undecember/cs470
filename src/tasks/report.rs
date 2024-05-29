use anyhow::Result;
use std::sync::{Arc, RwLock};
use std::time::Instant;

#[derive(Copy, Clone, Debug)]
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
pub struct TimingReportItem {
    token_index: usize,
    runner_type: RunnerType,
    action_type: ActionType,
    time_range: (Instant, Instant),
}

pub struct TaskReport {
    pub output_tokens: Vec<u32>,
    timings_report: RwLock<Vec<Arc<RwLock<TimingReportItem>>>>,
}

impl Default for TaskReport {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskReport {
    pub fn new() -> Self {
        Self {
            output_tokens: Vec::new(),
            timings_report: RwLock::new(Vec::new()),
        }
    }

    pub fn start(
        &self,
        runner_type: RunnerType,
        action_type: ActionType,
        token_index: usize,
    ) -> Arc<RwLock<TimingReportItem>> {
        let item = TimingReportItem {
            token_index,
            runner_type,
            action_type,
            time_range: (Instant::now(), Instant::now()),
        };
        let item = Arc::new(RwLock::new(item));
        self.timings_report.write().unwrap().push(item.clone());
        item
    }

    pub fn end(&self, item: Arc<RwLock<TimingReportItem>>) {
        item.write().unwrap().time_range.1 = Instant::now();
    }

    pub fn set_output_tokens(&mut self, tokens: &[u32]) {
        self.output_tokens.clear();
        self.output_tokens.extend_from_slice(tokens);
    }

    pub fn sort_timings(&mut self) {
        self.timings_report.write().unwrap().sort_by(|a, b| {
            a.read()
                .unwrap()
                .time_range
                .0
                .cmp(&b.read().unwrap().time_range.0)
        });
    }

    pub fn total_millis(&self) -> f64 {
        let timings_report_read = self.timings_report.read().unwrap();
        let mn = timings_report_read[0].read().unwrap().time_range.0;
        let mut mx = timings_report_read[0].read().unwrap().time_range.1;
        for i in 1..timings_report_read.len() {
            let t = timings_report_read[i].read().unwrap().time_range.1;
            if mx < t {
                mx = t;
            }
        }
        (mx - mn).as_micros() as f64 / 1000_f64
    }

    pub fn export_timings(&self, file_path: &str) -> Result<()> {
        let mut buf = String::new();

        let timings_report_read = self.timings_report.read().unwrap();
        let start_time = timings_report_read[0].read().unwrap().time_range.0;
        for item in timings_report_read.iter() {
            let item_read = item.read().unwrap();
            buf += format!("{} ", item_read.token_index).as_str();
            buf += match item_read.runner_type {
                RunnerType::Draft => "draft",
                RunnerType::Target => "target",
            };
            buf += match item_read.action_type {
                ActionType::ForwardKV => " forward_kv",
                ActionType::LogitsCalc => " logits_calc",
                ActionType::Sampling => " sampling",
            };
            buf += format!(
                " {} {}\n",
                (item_read.time_range.0 - start_time).as_micros(),
                (item_read.time_range.1 - start_time).as_micros(),
            )
            .as_str();
        }

        std::fs::write(file_path, buf.as_str())?;
        Ok(())
    }
}
