use super::T5Model;
use anyhow::{Error as E, Result};
use candle_core::Tensor;

impl T5Model {
    pub fn get_logits(
        &mut self,
        index: usize,
        decoder_tokens: &Tensor,
        encoder_output: &Tensor,
        output_tokens: &[u32],
    ) -> Result<Tensor> {
        let logits = self.runners[index]
            .decode(decoder_tokens, encoder_output)?
            .squeeze(0)?;
        if self.repeat_penalty == 1. {
            Ok(logits)
        } else {
            let start_at = output_tokens.len().saturating_sub(64);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                self.repeat_penalty,
                &output_tokens[start_at..],
            )
            .map_err(E::msg)
        }
    }
}
