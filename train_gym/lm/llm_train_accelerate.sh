export http_proxy="127.0.0.1:2334"
export https_proxy="127.0.0.1:2334"
export CUDA_VISIBLE_DEVICES=0
# python -m lang_mod_transformers.lang_mod_accelerate_simple \
python -m lang_mod_transformers.lang_mod_accelerate \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 14 \
    --per_device_eval_batch_size 14 \
    --do_train \
    --do_eval \
    --output_dir ./train_output \
    --report_to wandb \
    --block_size 1024 \
    --logging_steps 4 \
    --attn_implementation flash_attention_2 \
    --optimization_level opt_28 \
    --bf16 \
    --remove_unused_columns False \
    --gradient_checkpointing False \
    --num_train_epochs=5 \
    --save_steps 5000000 \
    --optim adamw_8bit


