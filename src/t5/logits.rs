use super::T5Model;
use anyhow::Result;
use candle_core::{DType, Tensor};
use rand::distributions::Distribution;
use rand::Rng;

impl<'g> T5Model<'g> {
    pub fn p_from_logits(&self, logits: &Tensor) -> Result<Vec<f32>> {
        let logits = logits.to_dtype(DType::F32)?;
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
