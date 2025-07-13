python -m lang_mod_transformers.accelerate_default_hf \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --output_dir ./train_output \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --block_size 1024 \