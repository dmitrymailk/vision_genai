pushd /code/

# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

# config_path=/code/train_gym/massive_train/evaluation/fsdp2_default_config.yaml
config_path=/code/train_gym/massive_train/evaluation/multi_gpu.yaml

accelerate launch --config_file $config_path -m train_gym.massive_train.evaluation.multigpu_lmeval