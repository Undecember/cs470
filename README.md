# Speculative Sampling

If you're using cs470 team server, check [this](docs/on_server.md).
You don't need anything below.

## Installation

### Cargo & Rust

```bash
curl https://sh.rustup.rs -sSf | sh
```

If you don't have `curl`,
```bash
sudo apt install curl
```

### CUDA

Install 
[12.1](https://developer.nvidia.com/cuda-12-1-0-download-archive) or
[12.4](https://developer.nvidia.com/cuda-12-4-0-download-archive)
from NVIDIA official website.

### libtorch

Install from [official website](https://pytorch.org/).
Scroll down and select Stable > Linux > LibTorch > C++/Java.
Select CUDA version that you installed and install libtorch.

## How to build

```bash
cargo build --release
```

## Usage

```bash
cargo run --release --bin cs470 -- --help
```
You can find input prompts [here](prompts/).

```bash
cargo run --release --bin cs470-batch -- --help
```
You can find batch test scenarios is [here](../scenarios/):
