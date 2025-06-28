## unsloth/Llama-3.2-1B-Instruct
### Speed up TODO
- liger cross entropy
- cut loss apple
- packed dataset, document attention
- torch compile + transformers models
- torch compile + torchtitan models

### Speed Up log

#### Default
```bash
python lang_mod_transformers.py \
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
    --logging_steps 8
```
```console
[00:54<09:29,  2.88it/s]
```

```bash
python lang_mod_transformers.py \
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
    --logging_steps 8
```
```console
CUDA out of memory.
```