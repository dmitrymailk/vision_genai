pushd /code/
export http_proxy="127.0.0.1:2334"
export https_proxy="127.0.0.1:2334"

export CUDA_VISIBLE_DEVICES=0
# export NCCL_CUMEM_HOST_ENABLE=0
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export CUBLAS_WORKSPACE_CONFIG=:4096:8
# export NCCL_DEBUG=TRACE
# export NCCL_DEBUG_SUBSYS=ALL
# export CUDA_LAUNCH_BLOCKING=1

# Accelerate configs
# config_path=/code/train_gym/distillation/sft/multi_gpu.yaml
config_path=/code/train_gym/distillation/sft/single_gpu.yaml

# train configs
# hf_train_config=/code/train_gym/distillation/sft/configs/sft_lora.yaml
hf_train_config=/code/train_gym/distillation/sft/configs/sft_lora_rmt.yaml

# start training
# accelerate launch --config_file $config_path -m \
python -m \
    train_gym.distillation.sft.sft_lora_train --config $hf_train_config > sft_lora_train_rmt.log 2>&1 &