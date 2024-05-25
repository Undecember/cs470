const INPUT_TEXT: &str = "sPummarize: Post-Traumatic Stress Disorder (PTSD) is a mental health condition triggered by experiencing or witnessing a traumatic event. It can develop after events such as natural disasters, serious accidents, terrorist acts, war or combat, rape, or other violent personal assaults. The disorder is characterized by intense, disturbing thoughts and feelings related to the experience that last long after the traumatic event has ended. People with PTSD may relive the event through flashbacks or nightmares; they may feel sadness, fear, or anger, and they may feel detached or estranged from other people.";

mod t5_test {
    use crate::INPUT_TEXT;
    use anyhow::{Error as E, Result};
    use candle_core::{Device, Tensor};
    use cs470::t5::*;
    use std::sync::Arc;

    #[test]
    fn t5_cache_consistency() -> Result<()> {
        const EPOCH: usize = 0xFF;
        const RUNNERS: usize = 5;

        let device = Arc::new(Device::new_cuda(0).map_err(E::msg)?);
        let model = T5ModelArgs {
            temperature: 1.0,
            seed: 299792458,
            top_p: None,
            no_kv_cache: false,
            repeat_penalty: 1.1,
        };
        let (mut model, mut tokenizer) = T5Model::new(
            ("google-t5/t5-3b".to_string(), "main".to_string()),
            device.clone(),
            model,
        )?;
        let tokenizer = tokenizer
            .with_padding(None)
            .with_truncation(None)
            .map_err(E::msg)?;
        let input_tokens = tokenizer
            .encode(INPUT_TEXT, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        let mut output_tokens = [model
            .config
            .decoder_start_token_id
            .unwrap_or(model.config.pad_token_id)
            as u32]
        .to_vec();
        let input_tokens = Tensor::new(input_tokens, &model.device)?.unsqueeze(0)?;
        model.init_runners(RUNNERS)?;
        let encoder_output = model.runners[0].encode(&input_tokens)?;
        for i in 0..EPOCH {
            let decoder_tokens = if model.config.use_cache {
                let last_token = *output_tokens.last().unwrap();
                Tensor::new(&[last_token], &model.device)?.unsqueeze(0)?
            } else {
                Tensor::new(output_tokens.as_slice(), &model.device)?.unsqueeze(0)?
            };
            let logits = model.get_logits(
                i % RUNNERS,
                &decoder_tokens,
                &encoder_output,
                &output_tokens,
            )?;
            let p = model.p_from_logits(&logits)?;
            let next_token_id = model.sample_from_p(&p)?;
            output_tokens.push(next_token_id);
            model.promote_runner(i % RUNNERS)?;
        }
        Ok(())
    }

    #[test]
    fn t5_cache_independency() -> Result<()> {
        const EPOCH: usize = 0xFF;
        const RUNNERS: usize = 5;

        let device = Arc::new(Device::new_cuda(0).map_err(E::msg)?);
        let model = T5ModelArgs {
            temperature: 1.0,
            seed: 299792458,
            top_p: None,
            no_kv_cache: false,
            repeat_penalty: 1.1,
        };
        let (mut model, mut tokenizer) = T5Model::new(
            ("google-t5/t5-3b".to_string(), "main".to_string()),
            device.clone(),
            model,
        )?;
        let tokenizer = tokenizer
            .with_padding(None)
            .with_truncation(None)
            .map_err(E::msg)?;
        let input_tokens = tokenizer
            .encode(INPUT_TEXT, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        let input_tokens = Tensor::new(input_tokens, &model.device)?.unsqueeze(0)?;
        model.init_runners(RUNNERS)?;
        let encoder_output = model.runners[0].encode(&input_tokens)?;
        model.promote_runner(0)?;
        for i in 0..RUNNERS {
            let mut output_tokens = [model
                .config
                .decoder_start_token_id
                .unwrap_or(model.config.pad_token_id)
                as u32]
            .to_vec();
            for _ in 0..EPOCH {
                let decoder_tokens = if model.config.use_cache {
                    let last_token = *output_tokens.last().unwrap();
                    Tensor::new(&[last_token], &model.device)?.unsqueeze(0)?
                } else {
                    Tensor::new(output_tokens.as_slice(), &model.device)?
                        .unsqueeze(0)?
                };
                let logits = model.get_logits(
                    i,
                    &decoder_tokens,
                    &encoder_output,
                    &output_tokens,
                )?;
                let p = model.p_from_logits(&logits)?;
                let next_token_id = model.sample_from_p(&p)?;
                output_tokens.push(next_token_id);
            }
        }
        Ok(())
    }
}
