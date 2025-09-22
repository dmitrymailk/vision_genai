# export CUDA_VISIBLE_DEVICES=0,1,2,3

model_name=unsloth/Llama-3.2-1B
accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=$model_name \
    --tasks arc_easy,hellaswag \
    --batch_size 32