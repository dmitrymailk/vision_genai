# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=2,3

model_name=unsloth/Llama-3.2-1B
accelerate launch --mixed_precision bf16 -m lm_eval \
    --model hf \
    --model_args pretrained=$model_name \
    --tasks squadv2 \
    --batch_size 64
    # --tasks arc_easy,hellaswag \
    # --batch_size 32


# 4 GPU A100, 05 min : 13 sec --batch_size 64
# | Tasks |Version|Filter|n-shot|   Metric   |   | Value |   |Stderr|
# |-------|------:|------|-----:|------------|---|------:|---|------|
# |squadv2|      3|none  |     0|HasAns_exact|↑  |20.5128|±  |   N/A|
# |       |       |none  |     0|HasAns_f1   |↑  |31.9434|±  |   N/A|
# |       |       |none  |     0|NoAns_exact |↑  | 0.0000|±  |   N/A|
# |       |       |none  |     0|NoAns_f1    |↑  | 0.0000|±  |   N/A|
# |       |       |none  |     0|best_exact  |↑  |50.0800|±  |   N/A|
# |       |       |none  |     0|best_f1     |↑  |50.0841|±  |   N/A|
# |       |       |none  |     0|exact       |↑  |10.2417|±  |   N/A|
# |       |       |none  |     0|f1          |↑  |15.9488|±  |   N/A|