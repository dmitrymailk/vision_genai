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
    --remove_unused_columns False \
    --gradient_checkpointing False

