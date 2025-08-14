
## unsloth/Llama-3.2-1B-Instruct
### Speed up TODO
- flash attn
- liger cross entropy
- cut loss apple
- packed dataset, document attention
- torch compile + transformers models
- torch compile + torchtitan models

### Speed Up log

#### Default
```bash
python -m lang_mod_transformers.lang_mod_transformers \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --bf16
```
```console
[00:17<15:36,  7.54it/s]
```
```bash
python -m lang_mod_transformers.lang_mod_transformers \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --bf16
```
```console
[00:20<11:49,  4.92it/s]
```
```bash
python -m lang_mod_transformers.lang_mod_transformers \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --bf16
```
```console
[00:16<10:12,  2.86it/s]
```

```bash
python -m lang_mod_transformers.lang_mod_transformers \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 5 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --bf16
```
```console
CUDA out of memory.
```
```bash
python -m lang_mod_transformers.lang_mod_transformers \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --bf16
```
```console
CUDA out of memory.
```

#### flash-attn 2.8.0.post2
```bash
python -m lang_mod_transformers.lang_mod_transformers \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --attn_implementation eager \
    --bf16
```
```console
CUDA out of memory.
```
```bash
python -m lang_mod_transformers.lang_mod_transformers \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --attn_implementation flash_attention_2 \
    --bf16
```
```console
[00:16<10:10,  2.87it/s]
```
```bash
python -m lang_mod_transformers.lang_mod_transformers \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --attn_implementation sdpa \
    --bf16
```
```console
[00:16<10:13,  2.85it/s]
```

#### Liger kernel
```bash
python -m lang_mod_transformers.lang_mod_transformers \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --attn_implementation flash_attention_2 \
    --optimization_level opt_2 \
    --bf16
```
```console
[00:18<09:34,  3.03it/s]
```
```bash
python -m lang_mod_transformers.lang_mod_transformers \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --attn_implementation flash_attention_2 \
    --optimization_level opt_3 \
    --bf16
```
```console
[00:16<09:02,  3.21it/s]
```
```bash
python -m lang_mod_transformers.lang_mod_transformers \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --attn_implementation flash_attention_2 \
    --optimization_level opt_4 \
    --bf16
```
```console
[00:18<09:33,  3.03it/s]
```
```bash
python -m lang_mod_transformers.lang_mod_transformers \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --attn_implementation flash_attention_2 \
    --optimization_level opt_5 \
    --bf16
```
```console
[00:19<09:51,  2.94it/s]
```


- вывод cross_entropy=True быстрее всех
- 2.86
- 2.94
- 3.21

#### apple cut-cross-entropy
- у них всех из коробки поломанные импорты
- https://github.com/apple/ml-cross-entropy
```bash
python -m lang_mod_transformers.lang_mod_transformers \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --attn_implementation flash_attention_2 \
    --optimization_level opt_6 \
    --bf16
```
```console
[00:17<09:24,  3.09it/s]
```

#### unsloth cut-cross-entropy
- https://github.com/unslothai/cut-cross-entropy
- переустанавливаем пакет от unsloth из сурсов
```bash
python -m lang_mod_transformers.lang_mod_transformers \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --attn_implementation flash_attention_2 \
    --optimization_level opt_7 \
    --bf16
```
```console
[00:25<09:10,  3.13it/s]
```
- в 1.076 быстрее при  per_device_train_batch_size 4
- в 1.168 быстрее при  per_device_train_batch_size 7

#### unsloth cut-cross-entropy+liger kernel
```bash
python -m lang_mod_transformers.lang_mod_transformers \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --attn_implementation flash_attention_2 \
    --optimization_level opt_9 \
    --bf16
```
```console
3.44it/s
```

- 3.44/2.86=1.202


#### transformers torch-compile
```bash
python -m lang_mod_transformers.lang_mod_transformers \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --attn_implementation flash_attention_2 \
    --optimization_level opt_10 \
    --bf16
```
```console
3.19it/s
```
- 3.19/2.86=1.115

#### transformers torch-compile+unsloth cut-cross-entropy+liger kernel
```bash
python -m lang_mod_transformers.lang_mod_transformers \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --attn_implementation flash_attention_2 \
    --optimization_level opt_11 \
    --bf16
```
```console
3.21it/s
```

#### torch.compile+liger kernel+liger cross entropy
```bash
python -m lang_mod_transformers.lang_mod_transformers \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --attn_implementation flash_attention_2 \
    --optimization_level opt_12 \
    --bf16 \
    --remove_unused_columns False
```
```console
3.02it/s
```

#### torch.compile+LlamaAttention
```bash
python -m lang_mod_transformers.lang_mod_transformers \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --attn_implementation flash_attention_2 \
    --optimization_level opt_13 \
    --bf16 \
    --remove_unused_columns False
```
```console
2.91it/s
```
#### torch.compile+LlamaDecoderLayer
```bash
python -m lang_mod_transformers.lang_mod_transformers \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --attn_implementation flash_attention_2 \
    --optimization_level opt_14 \
    --bf16 \
    --remove_unused_columns False
```
```console
3.13it/s
```
#### torch.compile+LlamaDecoderLayer+unsloth cut-cross-entropy+liger kernel
```bash
python -m lang_mod_transformers.lang_mod_transformers \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --attn_implementation flash_attention_2 \
    --optimization_level opt_15 \
    --bf16 \
    --remove_unused_columns False
```
```console
3.38it/s
```

#### torch.compile+mode="max-autotune"+LlamaDecoderLayer+unsloth cut-cross-entropy+liger kernel
```bash
python -m lang_mod_transformers.lang_mod_transformers \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --attn_implementation flash_attention_2 \
    --optimization_level opt_16 \
    --bf16 \
    --remove_unused_columns False
```
```console
3.59it/s
```
- 3.56/2.86=1.25

#### torch.compile+mode="max-autotune"+LlamaDecoderLayer
```bash
python -m lang_mod_transformers.lang_mod_transformers \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --attn_implementation flash_attention_2 \
    --optimization_level opt_17 \
    --bf16 \
    --remove_unused_columns False
```
```console
3.30it/s
```
#### torch.compile+mode="max-autotune"+LlamaDecoderLayer+unsloth cut-cross-entropy
```bash
python -m lang_mod_transformers.lang_mod_transformers \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --attn_implementation flash_attention_2 \
    --optimization_level opt_18 \
    --bf16 \
    --remove_unused_columns False
```
```console
3.66it/s
```
- требует адекватного warm up, чтобы не было рекомпиляции во время тренировки
- 3.66/2.86=1.279
- в 1.236 быстрее по общему времени
- в 1.37 быстрее при per_device_train_batch_size=8

#### torch.compile+mode="max-autotune"
```bash
python -m lang_mod_transformers.lang_mod_transformers \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --attn_implementation flash_attention_2 \
    --optimization_level opt_19 \
    --bf16 \
    --remove_unused_columns False
```
```console
3.38it/s
```
- 3.19 - no max autotune
- 3.38 - max-autotune 

#### torch.compile+mode="max-autotune"+LlamaDecoderLayer+unsloth cut-cross-entropy+cuda streams
```bash
python -m lang_mod_transformers.lang_mod_transformers \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --attn_implementation flash_attention_2 \
    --optimization_level opt_20 \
    --bf16 \
    --remove_unused_columns False
```
```console
3.66it/s
```

#### torch.compile+torch.ao.float8+LlamaDecoderLayer
```bash
python -m lang_mod_transformers.lang_mod_transformers \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --attn_implementation flash_attention_2 \
    --optimization_level opt_21 \
    --bf16 \
    --remove_unused_columns False \
    --gradient_checkpointing False \
    --num_train_epochs=3
```
```console
3.85it/s
```
- 

#### torch.compile+torch.ao.float8+LlamaDecoderLayer+unsloth cut-cross-entropy
```bash
python -m lang_mod_transformers.lang_mod_transformers \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --attn_implementation flash_attention_2 \
    --optimization_level opt_22 \
    --bf16 \
    --remove_unused_columns False \
    --gradient_checkpointing False \
    --num_train_epochs=3
```

```console
4.35it/s
```
- 4.35/2.86=1.52
- в 1.454 быстрее по общему времени
- в 1.603 быстрее при per_device_train_batch_size=8 (2.42it/s)
- в 1.725 быстрее при per_device_train_batch_size=12 (1.70it/s)
- в 1.762 быстрее при per_device_train_batch_size=14 (1.55it/s)

при использовании образа nvcr.io/nvidia/pytorch:25.06-py3
- default runtime 10m 40s
- при per_device_train_batch_size=14 (4.43it/s)
- в 1.8768 быстрее при per_device_train_batch_size=14 (1.60it/s)(5m 41s)
- в 1.9335 быстрее при per_device_train_batch_size=14 1.64it/s и bnb.optim.Adam8bit (5m 31s)
---
- 6 epochs, при per_device_train_batch_size=14 - 10m 42s, bnb.optim.Adam8bit
- 6 epochs, при per_device_train_batch_size=14 - 11m 06s torch.AdamW

#### accelerate.compile_regions+torch.ao.float8+LlamaDecoderLayer+unsloth cut-cross-entropy
```bash
python -m lang_mod_transformers.lang_mod_transformers \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --attn_implementation flash_attention_2 \
    --optimization_level opt_23 \
    --bf16 \
    --remove_unused_columns False \
    --gradient_checkpointing False \
    --num_train_epochs=3
```
```console
4.27it/s
```

#### accelerate transformer engine fp8 integration+unsloth cut-cross-entropy
- без cut-cross-entropy+per_device_train_batch_size=4 падает по памяти во время обучения

```bash
python -m lang_mod_transformers.lang_mod_accelerate \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --attn_implementation flash_attention_2 \
    --optimization_level opt_25 \
    --bf16 \
    --remove_unused_columns False \
    --gradient_checkpointing False \
    --num_train_epochs=3 \
```
```console
3.85it/s
```
- 3.85/2.86=1.34

- в 1.328 быстрее при per_device_train_batch_size 4
- в 1.43 быстрее при per_device_train_batch_size 7

#### accelerate transformer engine fp8 integration+unsloth cut-cross-entropy+torch.compile max autotune
```bash
python -m lang_mod_transformers.lang_mod_accelerate \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --attn_implementation flash_attention_2 \
    --optimization_level opt_26 \
    --bf16 \
    --remove_unused_columns False \
    --gradient_checkpointing False \
    --num_train_epochs=3 \
```
```
2.31it/s
```
#### accelerate transformer engine fp8 integration+unsloth cut-cross-entropy+torch.compile
```bash
python -m lang_mod_transformers.lang_mod_accelerate \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --attn_implementation flash_attention_2 \
    --optimization_level opt_26 \
    --bf16 \
    --remove_unused_columns False \
    --gradient_checkpointing False \
    --num_train_epochs=3 \
```

```console
2.45it/s
```
- в 1.6658 быстрее при per_device_train_batch_size 8

#### accelerate+torchtune+torch.compile
```bash
python -m lang_mod_transformers.lang_mod_accelerate \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --attn_implementation flash_attention_2 \
    --optimization_level opt_27 \
    --bf16 \
    --remove_unused_columns False \
    --gradient_checkpointing False \
    --num_train_epochs=3 \
    --save_steps 5000000
```
```console
3.23it/s
```
- при per_device_train_batch_size 4 - 3.23it/s (09:16)

#### accelerate+torchtune+torch.compile+float8
```bash
python -m lang_mod_transformers.lang_mod_accelerate \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --attn_implementation flash_attention_2 \
    --optimization_level opt_29 \
    --bf16 \
    --remove_unused_columns False \
    --gradient_checkpointing False \
    --num_train_epochs=3 \
    --save_steps 5000000
```
```console
4.14it/s
```
- при per_device_train_batch_size 4 - 4.14it/s 
- при per_device_train_batch_size 8 - 2.41it/s (06:41)
- при per_device_train_batch_size 10 - OOM

#### accelerate+torchtune+torch.compile+float8+unsloth cut-cross-entropy
```bash
python -m lang_mod_transformers.lang_mod_accelerate \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --attn_implementation flash_attention_2 \
    --optimization_level opt_30 \
    --bf16 \
    --remove_unused_columns False \
    --gradient_checkpointing False \
    --num_train_epochs=3 \
    --save_steps 5000000
```
```console
4.36it/s
```
- при per_device_train_batch_size 4 - 4.36it/s 
- при per_device_train_batch_size 8 - 2.57it/s (06:41)
- при per_device_train_batch_size 10 - 2.10it/s (05:56)
- при per_device_train_batch_size 14 - 1.58it/s (05:45)
- при per_device_train_batch_size 10 - 2.13it/s max-autotune,  
- при per_device_train_batch_size 14 - 1.63it/s при OptimizerInBackward

- в 1.8550 быстрее при per_device_train_batch_size 14. соответственно нет смысла использовать эту модель
- при per_device_train_batch_size 14 - 1.63it/s при OptimizerInBackward (05:34)
- в 1.9161 быстрее при per_device_train_batch_size 14 с [OptimizerInBackward](https://docs.pytorch.org/torchtune/main/tutorials/memory_optimizations.html#fusing-optimizer-step-into-backward-pass)
- при per_device_train_batch_size 14 - 1.64it/s при bnb.optim.Adam8bit (05:30)
- при per_device_train_batch_size 16 - 1.42it/s при bnb.optim.Adam8bit (05:40)
- при per_device_train_batch_size 17 - 1.33it/s  при bnb.optim.Adam8bit (05:38)
- при per_device_train_batch_size 14 - 1.64it/s при bnb.optim.Adam8bit+OptimizerInBackward
- при per_device_train_batch_size 14 - 1.57it/s при torch.optim.AdamW fused

#### accelerate+torchtune+torch.compile+int8+int4++unsloth cut-cross-entropy
- activation_config = FakeQuantizeConfig(torch.int8, "per_token", is_symmetric=False)
- weight_config = FakeQuantizeConfig(torch.int4, group_size=32)
```bash
python -m lang_mod_transformers.lang_mod_accelerate \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --attn_implementation flash_attention_2 \
    --optimization_level opt_31 \
    --bf16 \
    --remove_unused_columns False \
    --gradient_checkpointing False \
    --num_train_epochs=3 \
    --save_steps 5000000
```
```console
3.16it/s
```
- при per_device_train_batch_size 4 - 3.16it/s
- при per_device_train_batch_size 8 - OOM


#### accelerate+float8+torch.compile+unsloth cut cross entropy
```bash
export http_proxy="127.0.0.1:2334"
export https_proxy="127.0.0.1:2334"
# python -m lang_mod_transformers.lang_mod_accelerate_simple \
python -m lang_mod_transformers.lang_mod_accelerate \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 8 \
    --attn_implementation flash_attention_2 \
    --optimization_level opt_28 \
    --bf16 \
    --remove_unused_columns False \
    --gradient_checkpointing False \
    --num_train_epochs=5 \
    --save_steps 5000000 \
    --optim adamw_8bit
```

- batch 14 adam8bit [04:23<04:26,  1.44it/s] - ~23263 tok/sec
- batch 14 adamw 02:15<06:48,  1.59it/s - ~22932 tok/sec
- batch 5 no optimizations(opt_1) - ~12707 tok/sec

- 100_000_000_000/23263/60/60/24=49.75  дней, если тренить на одной 4090 1B модель

