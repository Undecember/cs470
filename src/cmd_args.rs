use crate::hf_models::t5::T5ModelArgs;
use clap::{Parser, ValueEnum};
use colored::Colorize;
use log::info;

#[derive(Clone, Debug, Copy, ValueEnum)]
pub enum WhichModel {
    #[value(name = "small")]
    T5Small,
    #[value(name = "base")]
    T5Base,
    #[value(name = "large")]
    T5Large,
    #[value(name = "3b")]
    T5_3B,
    #[value(name = "11b")]
    T5_11B,
    #[value(name = "fsmall")]
    FlanT5Small,
    #[value(name = "fbase")]
    FlanT5Base,
    #[value(name = "flarge")]
    FlanT5Large,
    #[value(name = "fxl")]
    FlanT5XLarge,
}

#[derive(Clone, Debug, Copy, ValueEnum)]
pub enum WhichPrefix {
    #[value(name = "summarize")]
    Summarize,
    #[value(name = "translate-german")]
    TranslateGerman,
    #[value(name = "translate-french")]
    TranslateFrench,
    #[value(name = "translate-romanian")]
    TranslateRomanian,
}

#[derive(Parser, Debug)]
#[group(required = true, multiple = false)]
pub struct PromptArgs {
    /// Prompt
    #[arg(short = 'p', long)]
    pub prompt: Option<String>,

    /// Prompt from file
    #[arg(long)]
    pub prompt_file: Option<String>,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// For quiet run
    #[arg(long, default_value = "false")]
    pub quiet: bool,

    #[clap(flatten)]
    pub prompt_group: PromptArgs,

    /// Prompt prefix type
    #[arg(long, default_value = "summarize")]
    pub prefix: WhichPrefix,

    /// Target model's repository path
    #[arg(short = 't', long, default_value = "3b")]
    pub target_model_repo: WhichModel,

    /// Draft model's repository path
    #[arg(short = 'd', long, default_value = "small")]
    pub draft_model_repo: WhichModel,

    /// Gamma value for speculative sampling
    #[arg(short = 'g', long, default_value_t = 5)]
    pub gamma: usize,

    /// Maximum number of tokens to generate
    #[arg(short = 'n', long, default_value_t = 1000)]
    pub max_tokens: usize,

    /// The temperature used to generate samples
    #[arg(long, default_value_t = 1.0)]
    pub temperature: f64,

    /// Random seed
    #[arg(long, default_value_t = 299792458)]
    pub seed: u64,

    /// Enable top-p sampling.
    #[arg(long)]
    pub top_p: Option<f64>,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    pub cpu: bool,

    /// Disable the key-value cache.
    #[arg(long)]
    pub no_kv_cache: bool,

    /// Repeat penalty
    #[arg(long, default_value_t = 1.1)]
    pub repeat_penalty: f64,

    /// Epsilon for smoothed KL divergence
    #[arg(long, default_value_t = 0.001)]
    pub kl_epsilon: f64,
}

impl Args {
    pub fn review(&self) {
        if let Some(file) = &self.prompt_group.prompt_file {
            info!("Prompt from file : {}", file.bold());
        }
        info!("Prefix : {}", self.get_prefix().bold());
        info!(
            "Target model : {}",
            Self::whichmodel_to_repo(self.target_model_repo).0.bold()
        );
        info!(
            "Draft model : {}",
            Self::whichmodel_to_repo(self.draft_model_repo).0.bold()
        );
        info!("Gamma : {}", self.gamma.to_string().bold());
        info!("Max tokens : {}", self.max_tokens.to_string().bold());
        info!(
            "Temperature : {}",
            format!("{:.3}", self.temperature).bold()
        );
        info!("Random seed : {}", self.seed.to_string().bold());
        info!(
            "Top p : {}",
            self.top_p
                .map_or("None".to_string(), |p| format!("{:.2}", p))
                .bold()
        );
        info!(
            "Running on device {}",
            if self.cpu { "CPU" } else { "CUDA" }.bold()
        );
        info!(
            "{} KV cache",
            if self.no_kv_cache { "No" } else { "Using" }.bold()
        );
        info!(
            "Repeat penalty : {}",
            format!("{:.2}", self.repeat_penalty).bold()
        );
        info!(
            "Epsilon (KL) : {}\n",
            format!("{:.3}", self.kl_epsilon).bold()
        );
    }

    pub fn get_prefix(&self) -> String {
        match self.prefix {
            WhichPrefix::Summarize => "summarize: ",
            WhichPrefix::TranslateGerman => "translate English to German: ",
            WhichPrefix::TranslateFrench => "translate English to French: ",
            WhichPrefix::TranslateRomanian => "translate English to Romanian: ",
        }
        .to_string()
    }

    pub fn get_draft_repo(&self) -> (String, String) {
        Self::whichmodel_to_repo(self.draft_model_repo)
    }

    pub fn get_target_repo(&self) -> (String, String) {
        Self::whichmodel_to_repo(self.target_model_repo)
    }

    pub fn get_model_args(&self) -> T5ModelArgs {
        T5ModelArgs {
            temperature: self.temperature,
            seed: self.seed,
            top_p: self.top_p,
            no_kv_cache: self.no_kv_cache,
            repeat_penalty: self.repeat_penalty,
        }
    }

    fn whichmodel_to_repo(which: WhichModel) -> (String, String) {
        let res = match which {
            WhichModel::T5Small => ("google-t5/t5-small", "main"),
            WhichModel::T5Base => ("google-t5/t5-base", "main"),
            WhichModel::T5Large => ("google-t5/t5-large", "main"),
            WhichModel::T5_3B => ("google-t5/t5-3b", "main"),
            WhichModel::T5_11B => ("google-t5/t5-11b", "refs/pr/6"),
            WhichModel::FlanT5Small => ("google/flan-t5-small", "main"),
            WhichModel::FlanT5Base => ("google/flan-t5-base", "main"),
            WhichModel::FlanT5Large => ("google/flan-t5-large", "main"),
            WhichModel::FlanT5XLarge => ("google/flan-t5-xl", "main"),
        };
        (res.0.to_string(), res.1.to_string())
    }
}

pub fn parse_args() -> Args {
    Args::parse()
}
