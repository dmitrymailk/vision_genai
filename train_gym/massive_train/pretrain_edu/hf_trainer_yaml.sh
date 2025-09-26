pushd /code/

# vpn for wandb
export http_proxy="127.0.0.1:2334"
export https_proxy="127.0.0.1:2334"

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0,1

# Accelerate configs
config_path=/code/train_gym/massive_train/pretrain_edu/fsdp2_default_config.yaml
# config_path=/code/train_gym/massive_train/pretrain_edu/multi_gpu.yaml
# train configs
hf_train_config=/code/train_gym/massive_train/pretrain_edu/configs/llama3.2-1B.yaml

# start training
accelerate launch --config_file $config_path -m \
    train_gym.massive_train.pretrain_edu.pretrain_hf_trainer --yaml_file $hf_train_config