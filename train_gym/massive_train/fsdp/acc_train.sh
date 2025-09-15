# export CUDA_VISIBLE_DEVICES=0,1
# export CUDA_VISIBLE_DEVICES=3,4
config_path=/code/train_gym/massive_train/fsdp/fsdp2_default_config.yaml
# accelerate launch --config_file $config_path accelerate_nlp_example.py

export http_proxy="127.0.0.1:2334"
export https_proxy="127.0.0.1:2334"
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=3
accelerate launch --config_file $config_path accelerate_nlp_example.py \
    --model_name_or_path=unsloth/Llama-3.2-1B-Instruct \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 52 \
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
    --dataloader_type mosaic_edu
    # --dataloader_type hf_edu
    # --dataloader_type ram_edu
    # --dataloader_type hf

# 23800 2 gpu, batch 24, 2048, A100-80GB, mosaic_edu
# 23586 4 gpu, batch 24, 2048, A100-80GB, mosaic_edu
# 23432 6 gpu, batch 24, 2048, A100-80GB, mosaic_edu

# 23446 6 gpu, batch 24, 1024, A100-80GB, mosaic_edu
# 24274 6 gpu, batch 52, 1024, A100-80GB, mosaic_edu max