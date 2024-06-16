use anyhow::{Error as E, Result};
use clap::ValueEnum;
use cs470::cmd_args::{Args, PromptArgs, WhichPrefix, WhichT5};
use cs470::hf_models::t5::T5ModelArgs;
use indicatif::{ProgressBar, ProgressStyle};
use serde::Deserialize;

pub fn deserialize_top_p<'de, D>(
    deserializer: D,
) -> std::result::Result<Vec<Option<f64>>, D::Error>
where
    D: serde::de::Deserializer<'de>,
{
    let mut res = Vec::<Option<f64>>::deserialize(deserializer)?;
    for top_p in res.iter_mut() {
        if *top_p == Some(1.0) {
            *top_p = None
        }
    }
    Ok(res)
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub prompt_path: String,
    pub prompt_cnt: usize,
    pub prefix: String,
    pub target_model_repo: String,
    pub draft_model_repo: String,
    pub gamma: Vec<usize>,
    pub adaptive_gamma: Vec<bool>,
    pub lenience: Vec<f64>,
    pub k_skipping: Vec<usize>,
    pub max_tokens: usize,
    pub early_reject_thr: Vec<f64>,
    pub temperature: Vec<f64>,
    pub seed: u64,
    #[serde(default, deserialize_with = "deserialize_top_p")]
    pub top_p: Vec<Option<f64>>,
    pub cpu: bool,
    pub no_kv_cache: bool,
    pub repeat_penalty: f64,
    pub kl_epsilon: Vec<f64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            prompt_path: "".to_string(),
            prompt_cnt: 10,
            prefix: "summarize".to_string(),
            target_model_repo: "large".to_string(),
            draft_model_repo: "small".to_string(),
            gamma: vec![5],
            adaptive_gamma: vec![false],
            lenience: vec![1.0],
            k_skipping: vec![1],
            max_tokens: 1000,
            early_reject_thr: vec![0.0],
            temperature: vec![1.0],
            seed: 299792458,
            top_p: vec![None],
            cpu: false,
            no_kv_cache: false,
            repeat_penalty: 1.1,
            kl_epsilon: vec![3e-7],
        }
    }
}

const PB_TEMPLATE: &str =
    "{spinner:.green} {pos}/{len} [{elapsed_precise}=>] [{bar:40.cyan/blue}] [=>{eta_precise}]";

impl Config {
    pub fn get_target_repo(&self) -> Result<(String, String)> {
        Ok(Args::which_t5_to_repo(
            WhichT5::from_str(self.target_model_repo.as_str(), true)
                .map_err(E::msg)?,
        ))
    }

    pub fn get_draft_repo(&self) -> Result<(String, String)> {
        Ok(Args::which_t5_to_repo(
            WhichT5::from_str(self.draft_model_repo.as_str(), true)
                .map_err(E::msg)?,
        ))
    }

    pub fn get_model_args(&self) -> T5ModelArgs {
        T5ModelArgs {
            temperature: self.temperature[0],
            seed: self.seed,
            top_p: self.top_p[0],
            no_kv_cache: self.no_kv_cache,
            repeat_penalty: self.repeat_penalty,
        }
    }

    pub fn iter(&self) -> Result<(ProgressBar, ConfigIter)> {
        let mut total_steps = self.prompt_cnt;
        total_steps *= self.gamma.len()
            * self.lenience.len()
            * self.k_skipping.len()
            * self.temperature.len()
            * self.top_p.len()
            * self.kl_epsilon.len();
        let pb = ProgressBar::new(total_steps as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(PB_TEMPLATE)?
                .progress_chars("#>-"),
        );
        let iter = ConfigIter {
            config: &self,
            need_init: true,
            prompt: 0,
            gamma: 0,
            adaptive_gamma: 0,
            lenience: 0,
            k_skipping: 0,
            early_reject_thr: 0,
            temperature: 0,
            top_p: 0,
            kl_epsilon: 0,
            end: false,
        };
        Ok((pb, iter))
    }
}

pub struct ConfigIter<'g> {
    config: &'g Config,
    need_init: bool,
    prompt: usize,
    gamma: usize,
    adaptive_gamma: usize,
    lenience: usize,
    k_skipping: usize,
    early_reject_thr: usize,
    temperature: usize,
    top_p: usize,
    kl_epsilon: usize,
    end: bool,
}

impl<'g> Iterator for ConfigIter<'g> {
    type Item = (Args, bool);
    fn next(&mut self) -> Option<Self::Item> {
        if self.end {
            return None;
        }
        let curr = Args {
            quiet: true,
            prompt_group: PromptArgs {
                prompt: None,
                prompt_file: Some(format!(
                    "{}{}.txt",
                    self.config.prompt_path, self.prompt
                )),
            },
            prefix: WhichPrefix::from_str(self.config.prefix.as_str(), true).unwrap(),
            target_model_repo: WhichT5::from_str(
                self.config.target_model_repo.as_str(),
                true,
            )
            .unwrap(),
            draft_model_repo: WhichT5::from_str(
                self.config.draft_model_repo.as_str(),
                true,
            )
            .unwrap(),
            gamma: self.config.gamma[self.gamma],
            adaptive_gamma: self.config.adaptive_gamma[self.adaptive_gamma],
            lenience: self.config.lenience[self.lenience],
            k_skipping: self.config.k_skipping[self.k_skipping],
            max_tokens: self.config.max_tokens,
            early_reject_thr: self.config.early_reject_thr[self.early_reject_thr],
            temperature: self.config.temperature[self.temperature],
            seed: self.config.seed,
            top_p: self.config.top_p[self.top_p],
            cpu: self.config.cpu,
            no_kv_cache: self.config.no_kv_cache,
            repeat_penalty: self.config.repeat_penalty,
            kl_epsilon: self.config.kl_epsilon[self.kl_epsilon],
        };
        let inc = |flag: &mut bool, c: &mut usize, c_max: &usize| {
            if !*flag {
                return;
            }
            *c += 1;
            if *c == *c_max {
                *c = 0;
            } else {
                *flag = false;
            }
        };
        let mut flag = true;
        inc(&mut flag, &mut self.prompt, &self.config.prompt_cnt);
        let need_init = self.need_init;
        self.need_init = flag;
        inc(&mut flag, &mut self.gamma, &self.config.gamma.len());
        inc(
            &mut flag,
            &mut self.adaptive_gamma,
            &self.config.adaptive_gamma.len(),
        );
        inc(&mut flag, &mut self.lenience, &self.config.lenience.len());
        inc(
            &mut flag,
            &mut self.k_skipping,
            &self.config.k_skipping.len(),
        );
        inc(
            &mut flag,
            &mut self.temperature,
            &self.config.temperature.len(),
        );
        inc(
            &mut flag,
            &mut self.early_reject_thr,
            &self.config.early_reject_thr.len(),
        );
        inc(&mut flag, &mut self.top_p, &self.config.top_p.len());
        inc(
            &mut flag,
            &mut self.kl_epsilon,
            &self.config.kl_epsilon.len(),
        );
        if flag {
            self.end = true;
        }
        Some((curr, need_init))
    }
}
