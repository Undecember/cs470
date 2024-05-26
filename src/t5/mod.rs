pub mod logits;
pub mod runner;

use anyhow::{Error as E, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use rand::SeedableRng;
use runner::T5Runner;
use std::sync::{Arc, RwLock};
use tokenizers::Tokenizer;

pub struct T5ModelArgs {
    pub temperature: f64,
    pub seed: u64,
    pub top_p: Option<f64>,
    pub no_kv_cache: bool,
    pub repeat_penalty: f32,
}

pub struct T5Model {
    pub device: Arc<Device>,
    pub rng: rand::rngs::StdRng,
    pub config: runner::Config,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub seed: u64,
    pub repeat_penalty: f32,
    pub runners: Vec<Arc<RwLock<T5Runner>>>,
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
        let weights_filename = vec![repo.get("model.safetensors")?];
        let config = std::fs::read_to_string(config_filename)?;
        let mut config: runner::Config = serde_json::from_str(&config)?;
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
        let runners = vec![Arc::new(RwLock::new(T5Runner::load(
            vb,
            &config,
            device.clone(),
            args.repeat_penalty,
        )?))];

        Ok((
            Self {
                device,
                rng,
                config,
                temperature: args.temperature,
                top_p: args.top_p,
                seed: args.seed,
                repeat_penalty: args.repeat_penalty,
                runners,
            },
            tokenizer,
        ))
    }

    pub fn init_runners(&mut self, cnt: usize) -> Result<()> {
        self.runners.truncate(1);
        self.runners[0].write().unwrap().clear_kv_cache();
        let kv_cache = self.runners[0].read().unwrap().export_kv_cache();
        for i in 1..cnt {
            let runner = T5Runner::clone(&*self.runners[0].read().unwrap());
            self.runners.push(Arc::new(RwLock::new(runner)));
            self.runners[i]
                .write()
                .unwrap()
                .import_kv_cache(&kv_cache)?;
        }
        Ok(())
    }

    pub fn propagate_kv_cache(&mut self, index: usize) -> Result<()> {
        let kv_cache = self.runners[index].read().unwrap().export_kv_cache();
        for i in 0..self.runners.len() {
            if i == index {
                continue;
            }
            self.runners[i]
                .write()
                .unwrap()
                .import_kv_cache(&kv_cache)?;
        }
        Ok(())
    }

    pub fn pass_kv_cache(&self, from: usize, to: usize) -> Result<()> {
        let kv_cache = self.runners[from].read().unwrap().export_kv_cache();
        self.runners[to].write().unwrap().import_kv_cache(&kv_cache)
    }
}
