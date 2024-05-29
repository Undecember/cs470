mod attention;
mod config;
mod layers;
pub mod logits;
pub mod runner;

pub use config::T5Config;

use anyhow::{Error as E, Result, bail};
use candle_core::{DType, Device, Result as CResult};
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
    pub config: T5Config,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub seed: u64,
    pub repeat_penalty: f32,
    pub runner: Arc<RwLock<T5Runner>>,
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
        let runner = Arc::new(RwLock::new(T5Runner::load(
            vb,
            &config,
            device.clone(),
            args.repeat_penalty,
        )?));

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
}

fn hub_load_safetensors(
    repo: &hf_hub::api::sync::ApiRepo,
    json_file: &str,
) -> Result<Vec<std::path::PathBuf>> {
    let json_file = repo.get(json_file).map_err(candle_core::Error::wrap)?;
    let json_file = std::fs::File::open(json_file)?;
    let json: serde_json::Value =
        serde_json::from_reader(&json_file).map_err(candle_core::Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => bail!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => bail!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }
    let safetensors_files = safetensors_files
        .iter()
        .map(|v| repo.get(v).map_err(candle_core::Error::wrap))
        .collect::<CResult<Vec<_>>>()?;
    Ok(safetensors_files)
}
