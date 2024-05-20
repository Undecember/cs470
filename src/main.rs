use anyhow::{Error as E, Result};
use clap::{Parser, ValueEnum};
use colored::Colorize;
use cs470::t5_model::T5Model;
use log::info;

#[derive(Clone, Debug, Copy, ValueEnum)]
enum WhichModel {
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
struct Args {
    /// Prompt to translate
    #[arg(short = 'p', long, required = true)]
    prompt: String,

    /// Target model's repository path.
    #[arg(short = 't', long, default_value = "3b")]
    target_model_repo: WhichModel,

    /// Draft model's repository path.
    #[arg(short = 'd', long, default_value = "small")]
    draft_model_repo: WhichModel,

    /// Maximum number of tokens to generate
    #[arg(short = 'n', long, default_value_t = 1000)]
    max_tokens: usize,

    /// The temperature used to generate samples
    #[arg(long, default_value_t = 1.0)]
    temperature: f64,

    /// Random seed
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// Enable top-p sampling.
    #[arg(long)]
    top_p: Option<f64>,

    /// Enable top-k sampling.
    #[arg(long)]
    top_k: Option<usize>,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Disable the key-value cache.
    #[arg(long)]
    no_kv_cache: bool,

    /// Repeat penalty
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,
}

fn main() -> Result<()> {
    std::env::set_var("RUST_LOG", "trace");
    env_logger::init();
    let args = Args::parse();

    if args.cpu {
        info!("Running on {}", "CPU".bold());
    } else {
        info!("Running on {}", "CUDA".bold());
    };

    info!(
        "Top k : {}",
        args.top_k.map_or("None".to_string(), |k| k.to_string())
    );
    info!(
        "Top p : {}",
        args.top_p.map_or("None".to_string(), |p| p.to_string())
    );

    let (draft_model_repo, draft_model_revision) =
        whichmodel_to_repo(args.draft_model_repo);
    let (target_model_repo, target_model_revision) =
        whichmodel_to_repo(args.target_model_repo);

    info!("Loading draft model {}...", draft_model_repo.bold());
    let (mut draft_model, _) = T5Model::load(
        draft_model_repo,
        draft_model_revision,
        args.no_kv_cache,
        args.temperature,
        args.cpu,
        args.top_p,
        args.seed,
        args.repeat_penalty,
    )?;
    info!("Loading target model {}...", target_model_repo.bold());
    let (mut target_model, mut tokenizer) = T5Model::load(
        target_model_repo,
        target_model_revision,
        args.no_kv_cache,
        args.temperature,
        args.cpu,
        args.top_p,
        args.seed,
        args.repeat_penalty,
    )?;

    let prompt = format!("summarize: {}", args.prompt);
    let tokenizer = tokenizer
        .with_padding(None)
        .with_truncation(None)
        .map_err(E::msg)?;
    let tokens = tokenizer
        .encode(prompt.clone(), true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();

    info!("Prompt : {}", prompt);
    info!("Start generating.");
    info!("[ {} ]\n", "Draft only".bold());
    let result = draft_model.single_sampling(&tokens, args.max_tokens)?;
    let dur = result.timings_report[result.timings_report.len() - 1].0
        - result.timings_report[0].0;
    info!(
        "Generation speed : {:.3} ms/tokens",
        dur.as_millis() as f64 / result.output_tokens.len() as f64
    );
    info!(
        "Generated text :\n{}",
        tokenizer
            .decode(&result.output_tokens, true)
            .map_err(E::msg)?
            .cyan()
    );

    info!("[ {} ]\n", "Target only".bold());
    let result = target_model.single_sampling(&tokens, args.max_tokens)?;
    let dur = result.timings_report[result.timings_report.len() - 1].0
        - result.timings_report[0].0;
    info!(
        "Generation speed : {:.3} ms/tokens",
        dur.as_millis() as f64 / result.output_tokens.len() as f64
    );
    info!(
        "Generated text :\n{}",
        tokenizer
            .decode(&result.output_tokens, true)
            .map_err(E::msg)?
            .cyan()
    );

    Ok(())
}

fn whichmodel_to_repo(which: WhichModel) -> (String, String) {
    let res = match which {
        WhichModel::T5Small => ("google-t5/t5-small", "main"),
        WhichModel::T5Base => ("google-t5/t5-base", "main"),
        WhichModel::T5Large => ("google-t5/t5-large", "main"),
        WhichModel::T5_3B => ("google-t5/t5-3b", "main"),
        WhichModel::T5_11B => ("google-t5/t5-11b", "refs/pr/6"),
    };
    (res.0.to_string(), res.1.to_string())
}
