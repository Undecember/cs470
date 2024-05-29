use super::attention::{T5LayerCrossAttention, T5LayerSelfAttention};
use super::layers::{T5LayerFF, T5LayerNorm};
use super::T5Config;

use anyhow::Result as AResult;
use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, VarBuilder};
use std::sync::Arc;

fn get_mask(size: usize, pad: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<_> = (pad..size + pad)
        .flat_map(|i| (0..size + pad).map(move |j| u8::from(j > i)))
        .collect();
    Tensor::from_slice(&mask, (size, size + pad), device)
}

#[derive(Debug, Clone)]
struct T5Block {
    self_attn: T5LayerSelfAttention,
    cross_attn: Option<T5LayerCrossAttention>,
    ff: T5LayerFF,
}

impl T5Block {
    fn rollback_kv_cache(&mut self, cnt: usize) -> AResult<()> {
        self.self_attn.rollback_kv_cache(cnt)?;
        if let Some(cross_attn) = &mut self.cross_attn {
            cross_attn.rollback_kv_cache(cnt)?;
        }
        Ok(())
    }
}

impl T5Block {
    fn load(
        has_relative_attention_bias: bool,
        decoder: bool,
        vb: VarBuilder,
        cfg: &T5Config,
    ) -> Result<Self> {
        let vb = vb.pp("layer");
        let self_attn = T5LayerSelfAttention::load(
            has_relative_attention_bias,
            decoder,
            vb.pp("0"),
            cfg,
        )?;
        let cross_attn = if cfg.is_decoder {
            Some(T5LayerCrossAttention::load(decoder, vb.pp("1"), cfg)?)
        } else {
            None
        };
        let ff_i = if cross_attn.is_some() { 2 } else { 1 };
        let ff = T5LayerFF::load(vb.pp(&ff_i.to_string()), cfg)?;
        Ok(Self {
            self_attn,
            cross_attn,
            ff,
        })
    }

    fn forward(
        &mut self,
        pad: usize,
        xs: &Tensor,
        position_bias: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let mask = match self.cross_attn.is_some() {
            true => {
                let mask_len = xs.dim(1)?;
                if mask_len <= 1 {
                    None
                } else {
                    Some(get_mask(mask_len, pad, xs.device())?)
                }
            }
            false => None,
        };
        let (mut xs, position_bias) =
            self.self_attn.forward(xs, position_bias, mask.as_ref())?;
        if let Some(cross_attn) = &mut self.cross_attn {
            (xs, _) =
                cross_attn.forward(&xs, None, encoder_hidden_states.unwrap())?;
        }
        let xs = self.ff.forward(&xs)?;
        Ok((xs, position_bias))
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
        self.cross_attn.iter_mut().for_each(|c| c.clear_kv_cache());
    }
}

#[derive(Debug, Clone)]
struct T5Stack {
    block: Vec<T5Block>,
    shared: Arc<Embedding>,
    final_layer_norm: T5LayerNorm,
}

impl T5Stack {
    fn rollback_kv_cache(&mut self, cnt: usize) -> AResult<()> {
        for b in &mut self.block {
            b.rollback_kv_cache(cnt)?;
        }
        Ok(())
    }
}

impl T5Stack {
    fn load(
        decoder: bool,
        vb: VarBuilder,
        shared: &Arc<Embedding>,
        cfg: &T5Config,
    ) -> Result<Self> {
        let block = (0..cfg.num_layers)
            .map(|i| {
                T5Block::load(i == 0, decoder, vb.pp(&format!("block.{i}")), cfg)
            })
            .collect::<Result<Vec<_>>>()?;
        let final_layer_norm = T5LayerNorm::load(
            cfg.d_model,
            cfg.layer_norm_epsilon,
            vb.pp("final_layer_norm"),
        )?;
        Ok(Self {
            block,
            shared: shared.clone(),
            final_layer_norm,
        })
    }

    fn forward(
        &mut self,
        pad: usize,
        input_ids: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<Tensor> {
        let input_embeds = self.shared.as_ref().forward(input_ids)?;
        let mut hidden_states = input_embeds;
        let mut position_bias = None;
        for block in self.block.iter_mut() {
            (hidden_states, position_bias) = block.forward(
                pad,
                &hidden_states,
                position_bias.as_ref(),
                encoder_hidden_states,
            )?
        }
        self.final_layer_norm.forward(&hidden_states)
    }

    fn clear_kv_cache(&mut self) {
        self.block.iter_mut().for_each(|b| b.clear_kv_cache())
    }
}

#[derive(Debug, Clone)]
pub struct T5Runner {
    encoder: T5Stack,
    decoder: T5Stack,
    d_model: usize,
    tie_word_embeddings: bool,
    lm_head: Option<Linear>,
    shared: Arc<Embedding>,
    device: Arc<Device>,
    use_cache: bool,
}

impl T5Runner {
    pub fn rollback_kv_cache(&mut self, cnt: usize) -> AResult<()> {
        self.encoder.rollback_kv_cache(cnt)?;
        self.decoder.rollback_kv_cache(cnt)?;
        Ok(())
    }
}

impl T5Runner {
    pub fn load(vb: VarBuilder, cfg: &T5Config, device: Arc<Device>) -> Result<Self> {
        assert!(cfg.is_encoder_decoder);
        let d_model = cfg.d_model;
        let shared_vb = if vb.contains_tensor("shared.weight") {
            vb.pp("shared")
        } else {
            vb.pp("decoder").pp("embed_tokens")
        };
        let shared = embedding(cfg.vocab_size, cfg.d_model, shared_vb)?;
        let shared = Arc::new(shared);

        let mut encoder_cfg = cfg.clone();
        encoder_cfg.is_decoder = false;
        encoder_cfg.use_cache = false;
        encoder_cfg.is_encoder_decoder = false;
        let encoder = T5Stack::load(false, vb.pp("encoder"), &shared, &encoder_cfg)?;

        let mut decoder_cfg = cfg.clone();
        decoder_cfg.is_decoder = true;
        decoder_cfg.is_encoder_decoder = false;
        decoder_cfg.num_layers = cfg.num_decoder_layers.unwrap_or(cfg.num_layers);
        let decoder = T5Stack::load(true, vb.pp("decoder"), &shared, &decoder_cfg)?;

        let tie_word_embeddings = cfg.tie_word_embeddings;
        let lm_head = if tie_word_embeddings {
            None
        } else {
            Some(linear_no_bias(
                cfg.d_model,
                cfg.vocab_size,
                vb.pp("lm_head"),
            )?)
        };

        Ok(Self {
            encoder,
            decoder,
            d_model,
            tie_word_embeddings,
            lm_head,
            shared,
            device,
            use_cache: cfg.use_cache,
        })
    }

    pub fn encode(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        self.encoder.forward(0, input_ids, None)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn clear_kv_cache(&mut self) {
        self.encoder.clear_kv_cache();
        self.decoder.clear_kv_cache();
    }

    pub fn forward_kv_cache(
        &mut self,
        range: std::ops::Range<usize>,
        encoder_output: &Tensor,
        output_tokens: &[u32],
    ) -> Result<Tensor> {
        let pad = range.start;
        let decoder_tokens = if self.use_cache {
            Tensor::new(&output_tokens[range], &self.device)?.unsqueeze(0)?
        } else {
            Tensor::new(&output_tokens[..range.end], &self.device)?.unsqueeze(0)?
        };
        self.decoder
            .forward(pad, &decoder_tokens, Some(encoder_output))
    }

    pub fn get_logits(&self, decoder_output: Tensor) -> Result<Tensor> {
        let scaling_factor = if self.tie_word_embeddings {
            (self.d_model as f64).sqrt()
        } else {
            1.0
        };
        let sequence_output = (decoder_output.squeeze(0)? * scaling_factor)?;
        match self.lm_head {
            None => sequence_output.matmul(&self.shared.embeddings().t()?),
            Some(ref lm_head) => lm_head.forward(&sequence_output),
        }
    }
}
