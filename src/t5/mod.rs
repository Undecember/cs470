pub mod logits;
pub mod runner;
pub mod sampling;

use anyhow::{Error as E, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use rand::SeedableRng;
use runner::T5Runner;
use tokenizers::Tokenizer;

pub struct T5ModelArgs {
    pub temperature: f64,
    pub seed: u64,
    pub top_p: Option<f64>,
    pub cpu: bool,
    pub no_kv_cache: bool,
    pub repeat_penalty: f32,
}

pub struct T5Model<'g> {
    pub device: Device,
    pub rng: rand::rngs::StdRng,
    pub config: runner::Config,
    pub temperature: f64,
    pub top_p: Option<f64>,
    pub seed: u64,
    pub repeat_penalty: f32,
    pub vb: VarBuilder<'g>,
    pub runners: Vec<T5Runner>,
}

impl<'g> T5Model<'g> {
    pub fn new(
        model_repo: (String, String),
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
                vb: vb.clone(),
                runners: Vec::<T5Runner>::new(),
            },
            tokenizer,
        ))
    }

    pub fn init_runners(&mut self, cnt: usize) -> Result<()> {
        while self.runners.len() < cnt {
            self.runners
                .push(T5Runner::load(self.vb.clone(), &self.config)?);
        }
        Ok(())
    }

    pub fn promote_runner(&mut self, index: usize) -> Result<()> {
        let kv_cache = self.runners[index].export_kv_cache()?;
        for runner in &mut self.runners {
            runner.import_kv_cache(&kv_cache)?;
        }
        Ok(())
    }
}
