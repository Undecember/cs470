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
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Prompt to translate
    #[arg(short = 'p', long, required = true)]
    pub prompt: String,

    /// Target model's repository path.
    #[arg(short = 't', long, default_value = "3b")]
    pub target_model_repo: WhichModel,

    /// Draft model's repository path.
    #[arg(short = 'd', long, default_value = "small")]
    pub draft_model_repo: WhichModel,

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
    pub repeat_penalty: f32,
}

pub fn parse_args() -> Args {
    Args::parse()
}

pub fn whichmodel_to_repo(which: WhichModel) -> (String, String) {
    let res = match which {
        WhichModel::T5Small => ("google-t5/t5-small", "main"),
        WhichModel::T5Base => ("google-t5/t5-base", "main"),
        WhichModel::T5Large => ("google-t5/t5-large", "main"),
        WhichModel::T5_3B => ("google-t5/t5-3b", "main"),
        WhichModel::T5_11B => ("google-t5/t5-11b", "refs/pr/6"),
    };
    (res.0.to_string(), res.1.to_string())
}

pub fn review_args(args: &Args) {
    info!(
        "Target model : {}",
        whichmodel_to_repo(args.target_model_repo).0.bold()
    );
    info!(
        "Draft model : {}",
        whichmodel_to_repo(args.draft_model_repo).0.bold()
    );
    info!("Max tokens : {}", args.max_tokens);
    info!("Temperature : {:.2}", args.temperature);
    info!("Random seed : {}", args.seed);
    info!(
        "Top p : {}",
        args.top_p.map_or("None".to_string(), |p| p.to_string())
    );
    info!(
        "Running on device {}",
        if args.cpu { "CPU" } else { "CUDA" }.bold()
    );
    info!(
        "{}sing KV cache",
        if args.no_kv_cache { "Not u" } else { "U" }
    );
    info!("Repeat_penalty : {:.2}", args.repeat_penalty);
}