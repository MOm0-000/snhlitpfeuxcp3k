---
library_name: peft
license: other
base_model: models/Qwen3-4B-Instruct-2507
tags:
- base_model:adapter:models/Qwen3-4B-Instruct-2507
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: qwen3-4b-qlora-dropout005-align100
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qwen3-4b-qlora-dropout005-align100

This model is a fine-tuned version of [models/Qwen3-4B-Instruct-2507](https://huggingface.co/models/Qwen3-4B-Instruct-2507) on the med_train_2k dataset.
It achieves the following results on the evaluation set:
- Loss: 1.3656

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 1
- eval_batch_size: 1
- seed: 23
- gradient_accumulation_steps: 16
- total_train_batch_size: 16
- optimizer: Use OptimizerNames.PAGED_ADAMW_8BIT with betas=(0.9,0.95) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 20
- training_steps: 100

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 1.3476        | 0.4   | 50   | 1.4026          |
| 1.3946        | 0.8   | 100  | 1.3656          |


### Framework versions

- PEFT 0.18.1
- Transformers 5.8.0
- Pytorch 2.11.0+cu128
- Datasets 4.0.0
- Tokenizers 0.22.2