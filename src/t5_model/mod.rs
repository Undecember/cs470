pub mod sampling;

use anyhow::{Error as E, Result};

use hf_hub::{api::sync::Api, Repo, RepoType};

use tokenizers::Tokenizer;

use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use candle_transformers::models::t5;
use t5::T5ForConditionalGeneration as T5ModelCG;

pub struct T5Model {
    pub model: T5ModelCG,
    pub device: Device,
    pub config: t5::Config,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub seed: u64,
    pub repeat_penalty: f32,
}

impl T5Model {
    pub fn load(
        model_repo: String,
        model_revision: String,
        no_kv_cache: bool,
        temperature: f64,
        cpu: bool,
        top_p: Option<f64>,
        seed: u64,
        repeat_penalty: f32,
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
        config.use_cache = !no_kv_cache;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let device = if cpu {
            Device::Cpu
        } else {
            Device::new_cuda(0).map_err(E::msg)?
        };
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weights_filename, DType::F32, &device)?
        };
        let model = t5::T5ForConditionalGeneration::load(vb, &config)?;

        Ok((
            Self {
                model,
                device,
                config,
                temperature,
                top_p,
                seed,
                repeat_penalty,
            },
            tokenizer,
        ))
    }
}
