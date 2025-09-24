pushd /code/

export CUDA_VISIBLE_DEVICES="0,1,2,3"
# export CUDA_VISIBLE_DEVICES="2,3"
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

config_path=/code/train_gym/massive_train/evaluation/fsdp2_default_config.yaml
# config_path=/code/train_gym/massive_train/evaluation/multi_gpu.yaml

accelerate launch --config_file $config_path -m train_gym.massive_train.evaluation.multigpu_lmeval # 07:47
# accelerate launch -m train_gym.massive_train.evaluation.multigpu_lmeval # 06:13

# DDP {'squadv2': {'alias': 'squadv2', 'exact,none': 10.13223279710267, 'exact_stderr,none': 'N/A', 'f1,none': 15.903001505345859, 'f1_stderr,none': 'N/A', 'HasAns_exact,none': 20.293522267206477, 'HasAns_exact_stderr,none': 'N/A', 'HasAns_f1,none': 31.85160878423944, 'HasAns_f1_stderr,none': 'N/A', 'NoAns_exact,none': 0.0, 'NoAns_exact_stderr,none': 'N/A', 'NoAns_f1,none': 0.0, 'NoAns_f1_stderr,none': 'N/A', 'best_exact,none': 50.08001347595385, 'best_exact_stderr,none': 'N/A', 'best_f1,none': 50.08221064234159, 'best_f1_stderr,none': 'N/A'}}
# =====
# =====
# =====
# FSDP {'squadv2': {'alias': 'squadv2', 'exact,none': 26.16019540133075, 'exact_stderr,none': 'N/A', 'f1,none': 26.20544278802512, 'f1_stderr,none': 'N/A', 'HasAns_exact,none': 0.0, 'HasAns_exact_stderr,none': 'N/A', 'HasAns_f1,none': 0.0906245314140051, 'HasAns_f1_stderr,none': 'N/A', 'NoAns_exact,none': 52.245584524810766, 'NoAns_exact_stderr,none': 'N/A', 'NoAns_f1,none': 52.245584524810766, 'NoAns_f1_stderr,none': 'N/A', 'best_exact,none': 50.07159100480081, 'best_exact_stderr,none': 'N/A', 'best_f1,none': 50.07159100480081, 'best_f1_stderr,none': 'N/A'}}
#============
#============
#============
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