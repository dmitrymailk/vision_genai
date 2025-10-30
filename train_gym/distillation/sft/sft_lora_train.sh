pushd /code/
export http_proxy="127.0.0.1:2334"
export https_proxy="127.0.0.1:2334"

export CUDA_VISIBLE_DEVICES=0

# Accelerate configs
config_path=/code/train_gym/distillation/sft/multi_gpu.yaml

# train configs
hf_train_config=/code/train_gym/distillation/sft/configs/sft_lora.yaml

# start training
accelerate launch --config_file $config_path -m \
    train_gym.distillation.sft.sft_lora_train --config $hf_train_config