# export CUDA_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=0
config_path=/code/train_gym/massive_train/fsdp/fsdp2_default_config.yaml
accelerate launch --config_file $config_path accelerate_nlp_example.py