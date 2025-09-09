export http_proxy="127.0.0.1:2334"
export https_proxy="127.0.0.1:2334"
# export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=3
python test_accelerate.py \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 50 \
    --per_device_eval_batch_size 4 \
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
    --num_train_epochs=1 \
    --save_steps 5000000 \
    --optim adamw_8bit \
    --dataloader_type hf_edu
    # --dataloader_type mosaic_edu
    # --dataloader_type hf

# 19563 - mosaic_edu 8 batch датасет состоял только из input_ids (неправильно так как attention masks и labels в некоторых случаях давали бы неверное вычисление)
# 19550 - hf_edu 8 batch, тоже самое. везде labels копировались прямо перед подачей в gpu

# 21433 - hf_edu 4 batch 2048, теперь заранее создаются attention masks, labels
# 21540 - mosaic_edu, тоже самое. как будто разницы вообще никакой

# 23763 - mosaic_edu 14 batch 1024
# 23393 - mosaic_edu 15 batch 1024
# 23758 - mosaic_edu 16 batch 1024
# 23764 - hf_edu 14 batch 1024 - никакой разницы :(

# 22600 - mosaic_edu 14 batch 1024, A100-80GB
# 24241 - mosaic_edu 32 batch 1024, A100-80GB
# 24130 - mosaic_edu 46 batch 1024, A100-80GB
# 24368 - mosaic_edu 50 batch 1024, A100-80GB

# 23392 - mosaic_edu 24 batch 2048, A100-80GB max
# 23242 - mosaic_edu 16 batch 2048, A100-80GB
# 23219 - mosaic_edu 18 batch 2048, A100-80GB

# 23656 - hf_edu 24 batch 2048 - еще и больше получилось, даже на большом батче
# 23694 - hf_edu 24 batch 1024
# 24604 - hf_edu 50 batch 1024