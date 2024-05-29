use super::T5Model;
use anyhow::{Error as E, Result};
use candle_core::{DType, Tensor};
use rand::distributions::Distribution;
use rand::Rng;

impl T5Model {
    pub fn p_from_logits(
        &self,
        logits: &Tensor,
        index: usize,
        output_tokens: &[u32],
    ) -> Result<Vec<f32>> {
        let logits = logits
            .get_on_dim(0, index)?
            .squeeze(0)?
            .to_dtype(DType::F32)?;
        let logits = if self.repeat_penalty == 1. {
            logits
        } else {
            let start_at = output_tokens.len().saturating_sub(64);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                self.repeat_penalty,
                &output_tokens[start_at..],
            )
            .map_err(E::msg)?
        };
        let logits = (&logits / self.temperature)?;
        let prs = candle_nn::ops::softmax_last_dim(&logits)?;
        let prs = prs.to_vec1()?;
        Ok(prs)
    }

    pub fn sample_from_p(&mut self, p: &Vec<f32>) -> Result<u32> {
        match self.top_p {
            Some(top_p) => {
                let mut p = p.clone();
                let mut argsort_indices = (0..p.len()).collect::<Vec<_>>();
                argsort_indices.sort_by(|&i, &j| p[j].total_cmp(&p[i]));
                let mut cumsum = 0.;
                for index in &argsort_indices {
                    if cumsum >= top_p as f32 {
                        p[*index] = 0.0;
                    } else {
                        cumsum += p[*index];
                    }
                }
                self.sample_multinomial(&p)
            }
            None => self.sample_multinomial(p),
        }
    }

    fn sample_multinomial(&mut self, p: &Vec<f32>) -> Result<u32> {
        let distr = rand::distributions::WeightedIndex::new(p)?;
        let next_token = distr.sample(&mut self.rng) as u32;
        Ok(next_token)
    }

    pub fn prob_test(&mut self, p: f32) -> bool {
        self.rng.gen_bool(p as f64)
    }
}
