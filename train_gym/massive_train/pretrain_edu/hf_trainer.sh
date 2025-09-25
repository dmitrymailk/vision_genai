pushd /code/

export http_proxy="127.0.0.1:2334"
export https_proxy="127.0.0.1:2334"

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0,1

config_path=/code/train_gym/massive_train/pretrain_edu/fsdp2_default_config.yaml
# config_path=/code/train_gym/massive_train/pretrain_edu/multi_gpu.yaml

accelerate launch --config_file $config_path -m train_gym.massive_train.pretrain_edu.pretrain_hf_trainer \
    --model_name_or_path unsloth/Llama-3.2-1B-Instruct \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 32 \
    --do_train \
    --do_eval \
    --output_dir ./hf_trainer_edu \
    --report_to wandb \
    --block_size 2048 \
    --logging_steps 4 \
    --eval_steps 500 \
    --eval_strategy steps \
    --include_num_input_tokens_seen \
    --attn_implementation flash_attention_2 \
    --optimization_level opt_1 \
    --bf16 \
    --remove_unused_columns False \
    --gradient_checkpointing False \
    --num_train_epochs=1 \
    --save_steps 5000000 \
    --data_seed 42 \
    --optim adamw_torch \
    --dataloader_type hf_edu
    # --dataloader_type mosaic_edu
    # --eval_on_start \
    # --dataloader_type ram_edu
    # --dataloader_type hf