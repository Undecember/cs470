use anyhow::Result;
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
    pub accept_report: Option<Vec<(u32, u32)>>,
    pub timings_report: Vec<TimingReportItem>,
    pub kl_divs: Option<(Vec<f64>, Vec<f64>)>,
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
            accept_report: None,
            timings_report: Vec::new(),
            kl_divs: None,
        }
    }

    pub fn start(
        &mut self,
        runner_type: RunnerType,
        action_type: ActionType,
        token_index: usize,
    ) {
        self.timings_report.push(TimingReportItem {
            token_index,
            runner_type,
            action_type,
            time_range: (Instant::now(), Instant::now()),
        });
    }

    pub fn end(&mut self) {
        self.timings_report.last_mut().unwrap().time_range.1 = Instant::now();
    }

    pub fn sort_timings(&mut self) {
        self.timings_report
            .sort_by(|a, b| a.time_range.0.cmp(&b.time_range.0));
    }

    pub fn total_millis(&self) -> f64 {
        let mn = self.timings_report[0].time_range.0;
        let mut mx = self.timings_report[0].time_range.1;
        for i in 1..self.timings_report.len() {
            let t = self.timings_report[i].time_range.1;
            if mx < t {
                mx = t;
            }
        }
        (mx - mn).as_micros() as f64 / 1000_f64
    }

    pub fn acceptance_rate(&self) -> Option<f64> {
        if let Some(report) = self.accept_report.as_ref() {
            let mut res = 0_f64;
            let mut cnt = 0_f64;
            for (a, g) in report {
                res += *a as f64 / *g as f64;
                cnt += 1_f64;
            }
            Some(res / cnt)
        } else {
            None
        }
    }

    pub fn export_timings(&self, file_path: &str) -> Result<()> {
        let mut buf = String::new();

        let start_time = self.timings_report[0].time_range.0;
        for item in self.timings_report.iter() {
            buf += format!("{} ", item.token_index).as_str();
            buf += match item.runner_type {
                RunnerType::Draft => "draft",
                RunnerType::Target => "target",
            };
            buf += match item.action_type {
                ActionType::ForwardKV => " forward_kv",
                ActionType::LogitsCalc => " logits_calc",
                ActionType::Sampling => " sampling",
            };
            buf += format!(
                " {} {}\n",
                (item.time_range.0 - start_time).as_micros(),
                (item.time_range.1 - start_time).as_micros(),
            )
            .as_str();
        }

        std::fs::write(file_path, buf.as_str())?;
        Ok(())
    }
}
