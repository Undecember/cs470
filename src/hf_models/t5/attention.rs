use super::layers::T5LayerNorm;
use super::T5Config;
use anyhow::Result as AResult;
use candle_core::{Module, Result, Tensor};
use candle_nn::{embedding, linear_no_bias, Embedding, Linear, VarBuilder};

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true =
        Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

#[derive(Debug, Clone)]
struct T5Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    n_heads: usize,
    d_kv: usize,
    relative_attention_bias: Option<Embedding>,
    relative_attention_num_buckets: usize,
    relative_attention_max_distance: usize,
    inner_dim: usize,
    use_cache: bool,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl T5Attention {
    fn rollback_kv_cache(&mut self, cnt: usize) -> AResult<()> {
        if let Some((k, v)) = &self.kv_cache {
            let kv_len = k.dim(2)? - cnt;
            let new_k = k.narrow(2, 0, kv_len)?;
            let new_v = v.narrow(2, 0, kv_len)?;
            self.kv_cache = Some((new_k, new_v));
        };
        Ok(())
    }
}

impl T5Attention {
    fn load(
        has_relative_attention_bias: bool,
        decoder: bool,
        vb: VarBuilder,
        cfg: &T5Config,
    ) -> Result<Self> {
        let inner_dim = cfg.num_heads * cfg.d_kv;
        let q = linear_no_bias(cfg.d_model, inner_dim, vb.pp("q"))?;
        let k = linear_no_bias(cfg.d_model, inner_dim, vb.pp("k"))?;
        let v = linear_no_bias(cfg.d_model, inner_dim, vb.pp("v"))?;
        let o = linear_no_bias(inner_dim, cfg.d_model, vb.pp("o"))?;
        let relative_attention_bias = if has_relative_attention_bias {
            let emb = embedding(
                cfg.relative_attention_num_buckets,
                cfg.num_heads,
                vb.pp("relative_attention_bias"),
            )?;
            Some(emb)
        } else {
            None
        };
        Ok(Self {
            q,
            k,
            v,
            o,
            n_heads: cfg.num_heads,
            d_kv: cfg.d_kv,
            relative_attention_bias,
            relative_attention_num_buckets: cfg.relative_attention_num_buckets,
            relative_attention_max_distance: cfg.relative_attention_max_distance,
            inner_dim,
            use_cache: cfg.use_cache && decoder,
            kv_cache: None,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        position_bias: Option<&Tensor>,
        key_value_states: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        // Performs Self-attention (if key_value_states is None) or attention
        // over source sentence (provided by key_value_states).
        let kv_input = match key_value_states {
            None => xs,
            Some(key_value_states) => key_value_states,
        };
        let (b_sz, q_len) = (xs.dim(0)?, xs.dim(1)?);
        let kv_len = kv_input.dim(1)?;
        let q = self.q.forward(xs)?;
        let k = self.k.forward(kv_input)?;
        let v = self.v.forward(kv_input)?;
        let q = q
            .reshape((b_sz, q_len, self.n_heads, self.d_kv))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut k = k
            .reshape((b_sz, kv_len, self.n_heads, self.d_kv))?
            .transpose(1, 2)?;
        let mut v = v
            .reshape((b_sz, kv_len, self.n_heads, self.d_kv))?
            .transpose(1, 2)?;

        if self.use_cache && key_value_states.is_none() {
            if let Some((kv_cache_k, kv_cache_v)) = &self.kv_cache {
                k = Tensor::cat(&[kv_cache_k, &k], 2)?;
                v = Tensor::cat(&[kv_cache_v, &v], 2)?;
            };
            self.kv_cache = Some((k.clone(), v.clone()));
        };
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        // TODO: Use flash_attn.
        let scores = { q.matmul(&k.t()?)? };
        let scores = match mask {
            None => scores,
            Some(mask) => masked_fill(
                &scores,
                &mask
                    .unsqueeze(0)?
                    .unsqueeze(0)?
                    .repeat((b_sz, self.n_heads))?,
                f32::NEG_INFINITY,
            )?,
        };

        let (scores, position_bias) = match position_bias {
            Some(position_bias) => (
                scores.broadcast_add(position_bias)?,
                Some(position_bias.clone()),
            ),
            None => {
                match &self.relative_attention_bias {
                    None => (scores, None),
                    Some(relative_attention_bias) => {
                        // This only handles the bidirectional case.
                        let kv_len = k.dim(2)?;
                        let (q_start, q_end) = match self.use_cache {
                            true => ((kv_len - q_len) as u32, kv_len as u32),
                            false => (0_u32, kv_len as u32),
                        };
                        let num_buckets =
                            self.relative_attention_num_buckets as u32 / 2;
                        let max_exact = num_buckets / 2;
                        let relative_position = (q_start..q_end)
                        .map(|i| {
                            (0..kv_len as u32)
                                .map(|j| {
                                    if i < j {
                                        if j - i < max_exact {
                                            j - i + num_buckets
                                        } else {
                                            let b = f32::log(
                                                (j - i) as f32 / max_exact as f32,
                                                self.relative_attention_max_distance as f32
                                                    / max_exact as f32,
                                            ) * (num_buckets - max_exact) as f32;
                                            u32::min(
                                                max_exact + num_buckets + b as u32,
                                                self.relative_attention_num_buckets as u32 - 1,
                                            )
                                        }
                                    } else if i - j < max_exact {
                                        i - j
                                    } else {
                                        let b = f32::log(
                                            (i - j) as f32 / max_exact as f32,
                                            self.relative_attention_max_distance as f32
                                                / max_exact as f32,
                                        ) * (num_buckets - max_exact) as f32;
                                        u32::min(max_exact + b as u32, num_buckets - 1)
                                    }
                                })
                                .collect::<Vec<u32>>()
                        })
                        .collect::<Vec<Vec<_>>>();
                        let relative_buckets =
                            Tensor::new(relative_position, q.device())?;
                        let position_bias = relative_attention_bias
                            .forward(&relative_buckets)?
                            .permute((2, 0, 1))?
                            .unsqueeze(0)?;
                        (scores.broadcast_add(&position_bias)?, Some(position_bias))
                        // TODO: position_bias_masked?
                    }
                }
            }
        };

        let attn_weights = { candle_nn::ops::softmax_last_dim(&scores)? };
        let attn_output = attn_weights.matmul(&v)?;
        let attn_output =
            attn_output
                .transpose(1, 2)?
                .reshape((b_sz, q_len, self.inner_dim))?;
        let attn_output = self.o.forward(&attn_output)?;
        Ok((attn_output, position_bias))
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

#[derive(Debug, Clone)]
pub struct T5LayerSelfAttention {
    self_attention: T5Attention,
    layer_norm: T5LayerNorm,
}

impl T5LayerSelfAttention {
    pub fn rollback_kv_cache(&mut self, cnt: usize) -> AResult<()> {
        self.self_attention.rollback_kv_cache(cnt)?;
        Ok(())
    }
}

impl T5LayerSelfAttention {
    pub fn load(h: bool, d: bool, vb: VarBuilder, cfg: &T5Config) -> Result<Self> {
        let self_attention = T5Attention::load(h, d, vb.pp("SelfAttention"), cfg)?;
        let layer_norm = T5LayerNorm::load(
            cfg.d_model,
            cfg.layer_norm_epsilon,
            vb.pp("layer_norm"),
        )?;
        Ok(Self {
            self_attention,
            layer_norm,
        })
    }

    pub fn forward(
        &mut self,
        xs: &Tensor,
        position_bias: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let normed_xs = self.layer_norm.forward(xs)?;
        let (ys, position_bias) =
            self.self_attention
                .forward(&normed_xs, position_bias, None, mask)?;
        let ys = (xs + ys)?;
        Ok((ys, position_bias))
    }

    pub fn clear_kv_cache(&mut self) {
        self.self_attention.clear_kv_cache()
    }
}

#[derive(Debug, Clone)]
pub struct T5LayerCrossAttention {
    cross_attention: T5Attention,
    layer_norm: T5LayerNorm,
}

impl T5LayerCrossAttention {
    pub fn rollback_kv_cache(&mut self, cnt: usize) -> AResult<()> {
        self.cross_attention.rollback_kv_cache(cnt)?;
        Ok(())
    }
}

impl T5LayerCrossAttention {
    pub fn load(decoder: bool, vb: VarBuilder, cfg: &T5Config) -> Result<Self> {
        let cross_attention =
            T5Attention::load(false, decoder, vb.pp("EncDecAttention"), cfg)?;
        let layer_norm = T5LayerNorm::load(
            cfg.d_model,
            cfg.layer_norm_epsilon,
            vb.pp("layer_norm"),
        )?;
        Ok(Self {
            cross_attention,
            layer_norm,
        })
    }

    pub fn forward(
        &mut self,
        hidden_states: &Tensor,
        position_bias: Option<&Tensor>,
        key_value_states: &Tensor,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let normed_hidden_states = self.layer_norm.forward(hidden_states)?;
        let (ys, position_bias) = self.cross_attention.forward(
            &normed_hidden_states,
            position_bias,
            Some(key_value_states),
            None,
        )?;
        let ys = (hidden_states + ys)?;
        Ok((ys, position_bias))
    }

    pub fn clear_kv_cache(&mut self) {
        self.cross_attention.clear_kv_cache()
    }
}
