use candle_nn::Activation;
use serde::Deserialize;

fn default_relative_attention_max_distance() -> usize {
    128
}

fn default_is_decoder() -> bool {
    false
}

fn default_use_cache() -> bool {
    true
}

fn default_tie_word_embeddings() -> bool {
    true
}

#[derive(Debug, Deserialize, Default, Clone, PartialEq)]
pub struct ActivationWithOptionalGating {
    pub gated: bool,
    pub activation: candle_nn::Activation,
}

pub fn deserialize_feed_forward_proj_activation<'de, D>(
    deserializer: D,
) -> std::result::Result<ActivationWithOptionalGating, D::Error>
where
    D: serde::de::Deserializer<'de>,
{
    match String::deserialize(deserializer)?.as_str() {
        "gated-gelu" => Ok(ActivationWithOptionalGating {
            gated: true,
            activation: candle_nn::Activation::NewGelu,
        }),
        "gated-silu" => Ok(ActivationWithOptionalGating {
            gated: true,
            activation: candle_nn::Activation::Silu,
        }),
        buf => {
            let activation =
                serde_plain::from_str(buf).map_err(serde::de::Error::custom)?;
            Ok(ActivationWithOptionalGating {
                gated: false,
                activation,
            })
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct T5Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_kv: usize,
    pub d_ff: usize,
    pub num_layers: usize,
    pub num_decoder_layers: Option<usize>,
    pub num_heads: usize,
    pub relative_attention_num_buckets: usize,
    #[serde(default = "default_relative_attention_max_distance")]
    pub relative_attention_max_distance: usize,
    pub dropout_rate: f64,
    pub layer_norm_epsilon: f64,
    pub initializer_factor: f64,
    #[serde(default, deserialize_with = "deserialize_feed_forward_proj_activation")]
    pub feed_forward_proj: ActivationWithOptionalGating,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_is_decoder")]
    pub is_decoder: bool,
    pub is_encoder_decoder: bool,
    #[serde(default = "default_use_cache")]
    pub use_cache: bool,
    pub pad_token_id: usize,
    pub eos_token_id: usize,
    pub decoder_start_token_id: Option<usize>,
}

impl Default for T5Config {
    fn default() -> Self {
        Self {
            vocab_size: 32128,
            d_model: 512,
            d_kv: 64,
            d_ff: 2048,
            num_layers: 6,
            num_decoder_layers: None,
            num_heads: 8,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            dropout_rate: 0.1,
            layer_norm_epsilon: 1e-6,
            initializer_factor: 1.0,
            feed_forward_proj: ActivationWithOptionalGating {
                gated: false,
                activation: Activation::Relu,
            },
            tie_word_embeddings: true,
            is_decoder: false,
            is_encoder_decoder: true,
            use_cache: true,
            pad_token_id: 0,
            eos_token_id: 1,
            decoder_start_token_id: Some(0),
        }
    }
}
