pub mod logits;
pub mod sampling;

use crate::cmd_args::Args;
use anyhow::{Error as E, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::t5;
use hf_hub::{api::sync::Api, Repo, RepoType};
use rand::SeedableRng;
use t5::T5ForConditionalGeneration as T5ModelCG;
use tokenizers::Tokenizer;

pub struct T5Model {
    pub device: Device,
    pub rng: rand::rngs::StdRng,
    pub config: t5::Config,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub seed: u64,
    pub repeat_penalty: f32,
    pub cgs: Vec<T5ModelCG>,
}

impl T5Model {
    pub fn new(
        model_repo: String,
        model_revision: String,
        args: &Args,
    ) -> Result<(Self, Tokenizer)> {
        let repo =
            Repo::with_revision(model_repo.clone(), RepoType::Model, model_revision);
        let api = Api::new()?;
        let repo = api.repo(repo);
        let config_filename = repo.get("config.json")?;
        let tokenizer_filename = repo.get("tokenizer.json")?;
        let weights_filename = vec![repo.get("model.safetensors")?];
        let config = std::fs::read_to_string(config_filename)?;
        let mut config: t5::Config = serde_json::from_str(&config)?;
        config.use_cache = !args.no_kv_cache;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let device = if args.cpu {
            Device::Cpu
        } else {
            Device::new_cuda(0).map_err(E::msg)?
        };
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &weights_filename,
                DType::F32,
                &device,
            )?
        };
        let cgs = vec![T5ModelCG::load(vb, &config)?];
        let rng = rand::rngs::StdRng::seed_from_u64(args.seed);

        Ok((
            Self {
                device,
                rng,
                config,
                temperature: args.temperature,
                top_p: args.top_p,
                seed: args.seed,
                repeat_penalty: args.repeat_penalty,
                cgs,
            },
            tokenizer,
        ))
    }

    pub fn init_cgs(&mut self, cnt: usize) {
        self.cgs.truncate(1);
        for _ in 1..cnt {
            self.cgs.push(self.cgs[0].clone());
        }
    }

    pub fn promote_cg(&mut self, index: usize) {
        self.cgs.drain(..index);
        self.cgs.truncate(1);
    }
}
