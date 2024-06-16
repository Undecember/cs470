use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Scenario file name
    #[arg(short = 's', long, required = true)]
    pub scenario: String,

    /// Result file name
    #[arg(short = 'o', long, required = true)]
    pub output: String,
}

pub fn parse_args() -> Args {
    Args::parse()
}
