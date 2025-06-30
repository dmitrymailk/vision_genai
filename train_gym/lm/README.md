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
    --optimization_level opt_6 \
    --bf16
```
```console
[00:25<09:10,  3.13it/s]
```

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
3.67it/s
```

- 3.67/2.86=1.283

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
