

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0,1

config_path=/code/train_gym/massive_train/pretrain_edu/fsdp2_default_config.yaml
# config_path=/code/train_gym/massive_train/pretrain_edu/multi_gpu.yaml

accelerate launch --config_file $config_path test_dist_dataloader.py