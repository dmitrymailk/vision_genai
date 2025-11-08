pushd /code/
export http_proxy="127.0.0.1:2334"
export https_proxy="127.0.0.1:2334"

export CUDA_VISIBLE_DEVICES=0

# Accelerate configs
config_path=/code/train_gym/distillation/sft/single_gpu.yaml

# train configs
hf_train_config=/code/train_gym/distillation/sft/configs/sft_full_rmt.yaml

# start training
# accelerate launch --config_file $config_path -m \
python -m train_gym.distillation.sft.sft_full_train --config $hf_train_config