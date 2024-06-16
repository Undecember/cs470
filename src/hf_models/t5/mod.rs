mod attention;
mod config;
mod layers;
pub mod logits;
pub mod runner;

pub use config::T5Config;

use super::hub_load_safetensors;
use anyhow::{Error as E, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use rand::SeedableRng;
use runner::T5Runner;
use std::sync::Arc;
use tokenizers::Tokenizer;

#[derive(Debug)]
pub struct T5ModelArgs {
    pub temperature: f64,
    pub seed: u64,
    pub top_p: Option<f64>,
    pub no_kv_cache: bool,
    pub repeat_penalty: f64,
}

pub struct T5Model {
    pub device: Arc<Device>,
    pub rng: rand::rngs::StdRng,
    pub config: T5Config,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub seed: u64,
    pub repeat_penalty: f64,
    pub runner: T5Runner,
}

impl T5Model {
    pub fn new(
        model_repo: (String, String),
        device: Arc<Device>,
        args: T5ModelArgs,
    ) -> Result<(Self, Tokenizer)> {
        let repo =
            Repo::with_revision(model_repo.0.clone(), RepoType::Model, model_repo.1);
        let api = Api::new()?;
        let repo = api.repo(repo);
        let config_filename = repo.get("config.json")?;
        let tokenizer_filename = repo.get("tokenizer.json")?;
        let weights_filename = if model_repo.0 == "google/flan-t5-xl" {
            hub_load_safetensors(&repo, "model.safetensors.index.json")?
        } else {
            vec![repo.get("model.safetensors")?]
        };
        let config = std::fs::read_to_string(config_filename)?;
        let mut config: T5Config = serde_json::from_str(&config)?;
        config.use_cache = !args.no_kv_cache;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &weights_filename,
                DType::F32,
                &device,
            )?
        };
        let rng = rand::rngs::StdRng::seed_from_u64(args.seed);
        let runner = T5Runner::load(vb, &config, device.clone())?;

        Ok((
            Self {
                device,
                rng,
                config,
                temperature: args.temperature,
                top_p: args.top_p,
                seed: args.seed,
                repeat_penalty: args.repeat_penalty,
                runner,
            },
            tokenizer,
        ))
    }

    pub fn reset_rng(&mut self) {
        self.rng = rand::rngs::StdRng::seed_from_u64(self.seed);
    }

    pub fn set_model_args(&mut self, args: T5ModelArgs) {
        self.reset_rng();
        self.temperature = args.temperature;
        self.top_p = args.top_p;
        self.seed = args.seed;
        self.repeat_penalty = args.repeat_penalty;
    }
}
