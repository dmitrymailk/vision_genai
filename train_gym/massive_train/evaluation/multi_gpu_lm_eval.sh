# export CUDA_VISIBLE_DEVICES=0,1,2,3
# # export CUDA_VISIBLE_DEVICES=2,3

# model_name=unsloth/Llama-3.2-1B
# accelerate launch --mixed_precision bf16 -m lm_eval \
#     --model hf \
#     --model_args pretrained=$model_name \
#     --tasks arc_easy,hellaswag,winogrande,sciq,copa,openbookqa,mmlu_stem,mmlu_other,mmlu_social_sciences,mmlu_humanities \
#     --batch_size 32
    # --tasks arc_easy,hellaswag \
    # --batch_size 32


# 4 GPU A100, 01:15 --batch_size 32
# hf (pretrained=unsloth/Llama-3.2-1B), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 32
# |                Tasks                 |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
# |--------------------------------------|------:|------|-----:|--------|---|-----:|---|-----:|
# |arc_easy                              |      1|none  |     0|acc     |↑  |0.6532|±  |0.0098|
# |                                      |       |none  |     0|acc_norm|↑  |0.6065|±  |0.0100|
# |copa                                  |      1|none  |     0|acc     |↑  |0.7700|±  |0.0423|
# |hellaswag                             |      1|none  |     0|acc     |↑  |0.4772|±  |0.0050|
# |                                      |       |none  |     0|acc_norm|↑  |0.6377|±  |0.0048|
# |humanities                            |      2|none  |      |acc     |↑  |0.3441|±  |0.0068|
# | - formal_logic                       |      1|none  |     0|acc     |↑  |0.2698|±  |0.0397|
# | - high_school_european_history       |      1|none  |     0|acc     |↑  |0.4788|±  |0.0390|
# | - high_school_us_history             |      1|none  |     0|acc     |↑  |0.4412|±  |0.0348|
# | - high_school_world_history          |      1|none  |     0|acc     |↑  |0.4262|±  |0.0322|
# | - international_law                  |      1|none  |     0|acc     |↑  |0.5289|±  |0.0456|
# | - jurisprudence                      |      1|none  |     0|acc     |↑  |0.4537|±  |0.0481|
# | - logical_fallacies                  |      1|none  |     0|acc     |↑  |0.3313|±  |0.0370|
# | - moral_disputes                     |      1|none  |     0|acc     |↑  |0.3613|±  |0.0259|
# | - moral_scenarios                    |      1|none  |     0|acc     |↑  |0.2391|±  |0.0143|
# | - philosophy                         |      1|none  |     0|acc     |↑  |0.4244|±  |0.0281|
# | - prehistory                         |      1|none  |     0|acc     |↑  |0.4290|±  |0.0275|
# | - professional_law                   |      1|none  |     0|acc     |↑  |0.2947|±  |0.0116|
# | - world_religions                    |      1|none  |     0|acc     |↑  |0.5029|±  |0.0383|
# |other                                 |      2|none  |      |acc     |↑  |0.4104|±  |0.0088|
# | - business_ethics                    |      1|none  |     0|acc     |↑  |0.3300|±  |0.0473|
# | - clinical_knowledge                 |      1|none  |     0|acc     |↑  |0.3660|±  |0.0296|
# | - college_medicine                   |      1|none  |     0|acc     |↑  |0.3468|±  |0.0363|
# | - global_facts                       |      1|none  |     0|acc     |↑  |0.3100|±  |0.0465|
# | - human_aging                        |      1|none  |     0|acc     |↑  |0.4170|±  |0.0331|
# | - management                         |      1|none  |     0|acc     |↑  |0.4272|±  |0.0490|
# | - marketing                          |      1|none  |     0|acc     |↑  |0.4957|±  |0.0328|
# | - medical_genetics                   |      1|none  |     0|acc     |↑  |0.4500|±  |0.0500|
# | - miscellaneous                      |      1|none  |     0|acc     |↑  |0.4764|±  |0.0179|
# | - nutrition                          |      1|none  |     0|acc     |↑  |0.4281|±  |0.0283|
# | - professional_accounting            |      1|none  |     0|acc     |↑  |0.2801|±  |0.0268|
# | - professional_medicine              |      1|none  |     0|acc     |↑  |0.3897|±  |0.0296|
# | - virology                           |      1|none  |     0|acc     |↑  |0.4036|±  |0.0382|
# |social sciences                       |      2|none  |      |acc     |↑  |0.3991|±  |0.0087|
# | - econometrics                       |      1|none  |     0|acc     |↑  |0.2193|±  |0.0389|
# | - high_school_geography              |      1|none  |     0|acc     |↑  |0.5051|±  |0.0356|
# | - high_school_government_and_politics|      1|none  |     0|acc     |↑  |0.4560|±  |0.0359|
# | - high_school_macroeconomics         |      1|none  |     0|acc     |↑  |0.3205|±  |0.0237|
# | - high_school_microeconomics         |      1|none  |     0|acc     |↑  |0.2899|±  |0.0295|
# | - high_school_psychology             |      1|none  |     0|acc     |↑  |0.4440|±  |0.0213|
# | - human_sexuality                    |      1|none  |     0|acc     |↑  |0.4733|±  |0.0438|
# | - professional_psychology            |      1|none  |     0|acc     |↑  |0.3546|±  |0.0194|
# | - public_relations                   |      1|none  |     0|acc     |↑  |0.3636|±  |0.0461|
# | - security_studies                   |      1|none  |     0|acc     |↑  |0.3755|±  |0.0310|
# | - sociology                          |      1|none  |     0|acc     |↑  |0.5672|±  |0.0350|
# | - us_foreign_policy                  |      1|none  |     0|acc     |↑  |0.5400|±  |0.0501|
# |stem                                  |      2|none  |      |acc     |↑  |0.3216|±  |0.0082|
# | - abstract_algebra                   |      1|none  |     0|acc     |↑  |0.2500|±  |0.0435|
# | - anatomy                            |      1|none  |     0|acc     |↑  |0.4963|±  |0.0432|
# | - astronomy                          |      1|none  |     0|acc     |↑  |0.3947|±  |0.0398|
# | - college_biology                    |      1|none  |     0|acc     |↑  |0.3819|±  |0.0406|
# | - college_chemistry                  |      1|none  |     0|acc     |↑  |0.2900|±  |0.0456|
# | - college_computer_science           |      1|none  |     0|acc     |↑  |0.3800|±  |0.0488|
# | - college_mathematics                |      1|none  |     0|acc     |↑  |0.2900|±  |0.0456|
# | - college_physics                    |      1|none  |     0|acc     |↑  |0.2451|±  |0.0428|
# | - computer_security                  |      1|none  |     0|acc     |↑  |0.5200|±  |0.0502|
# | - conceptual_physics                 |      1|none  |     0|acc     |↑  |0.3532|±  |0.0312|
# | - electrical_engineering             |      1|none  |     0|acc     |↑  |0.4069|±  |0.0409|
# | - elementary_mathematics             |      1|none  |     0|acc     |↑  |0.2540|±  |0.0224|
# | - high_school_biology                |      1|none  |     0|acc     |↑  |0.3935|±  |0.0278|
# | - high_school_chemistry              |      1|none  |     0|acc     |↑  |0.2709|±  |0.0313|
# | - high_school_computer_science       |      1|none  |     0|acc     |↑  |0.3500|±  |0.0479|
# | - high_school_mathematics            |      1|none  |     0|acc     |↑  |0.2185|±  |0.0252|
# | - high_school_physics                |      1|none  |     0|acc     |↑  |0.2185|±  |0.0337|
# | - high_school_statistics             |      1|none  |     0|acc     |↑  |0.2407|±  |0.0292|
# | - machine_learning                   |      1|none  |     0|acc     |↑  |0.3571|±  |0.0455|
# |openbookqa                            |      1|none  |     0|acc     |↑  |0.2660|±  |0.0198|
# |                                      |       |none  |     0|acc_norm|↑  |0.3700|±  |0.0216|
# |sciq                                  |      1|none  |     0|acc     |↑  |0.9110|±  |0.0090|
# |                                      |       |none  |     0|acc_norm|↑  |0.8840|±  |0.0101|
# |winogrande                            |      1|none  |     0|acc     |↑  |0.6069|±  |0.0137|

# |    Groups     |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
# |---------------|------:|------|------|------|---|-----:|---|-----:|
# |humanities     |      2|none  |      |acc   |↑  |0.3441|±  |0.0068|
# |other          |      2|none  |      |acc   |↑  |0.4104|±  |0.0088|
# |social sciences|      2|none  |      |acc   |↑  |0.3991|±  |0.0087|
# |stem           |      2|none  |      |acc   |↑  |0.3216|±  |0.0082|

###########
###########
###########
########### babilong
###########

export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=0

model_name=HuggingFaceTB/SmolLM2-360M
# model_name=unsloth/Llama-3.2-1B-Instruct
# model_name=Qwen/Qwen2.5-7B-Instruct
# lm_eval \
    # --model_args pretrained=$model_name \
accelerate launch --mixed_precision bf16 -m lm_eval \
    --model hf \
    --tasks babilongv2_qa1_under_4k_base,babilongv2_qa2_under_4k_base,babilongv2_qa3_under_4k_base,babilongv2_qa4_under_4k_base,babilongv2_qa5_under_4k_base \
    --model_args pretrained=$model_name \
    --batch_size 8 \
    # --apply_chat_template \
    # --system_instruction "You are a helpful assistant." \
    # --tasks babilongv2_qa1_under_4k_instruct,babilongv2_qa2_under_4k_instruct,babilongv2_qa3_under_4k_instruct,babilongv2_qa4_under_4k_instruct,babilongv2_qa5_under_4k_instruct \
    # --tasks babilong_qa1,babilong_qa2,babilong_qa3,babilong_qa4,babilong_qa5 \
    # --tasks babilongv2_qa2_under_4k_instruct \
    # --output_path chat_template_test_results.json \
    # --log_samples \
    # --tasks babilongv2_qa2_0k_instruct \
    # --batch_size 96
    # --tasks babilongv2_qa2_0k_base,babilongv2_qa2_1k_base,babilongv2_qa2_2k_base,babilongv2_qa2_4k_base \
    # --tasks babilongv2_qa1_0k_base,babilongv2_qa1_1k_base,babilongv2_qa1_2k_base,babilongv2_qa1_4k_base \


# 4gpu batch 96 HuggingFaceTB/SmolLM2-360M - 3:33 - babilongv2_qa1_under_4k_base,babilongv2_qa2_under_4k_base,babilongv2_qa3_under_4k_base,babilongv2_qa4_under_4k_base,babilongv2_qa5_under_4k_base
# 4gpu batch 32 unsloth/Llama-3.2-3B-Instruct - 11:19 - babilongv2_qa1_under_4k_instruct,babilongv2_qa2_under_4k_instruct,babilongv2_qa3_under_4k_instruct,babilongv2_qa4_under_4k_instruct,babilongv2_qa5_under_4k_instruct
# 4gpu batch 64 unsloth/Llama-3.2-1B-Instruct - 04:17 - babilongv2_qa1_under_4k_instruct,babilongv2_qa2_under_4k_instruct,babilongv2_qa3_under_4k_instruct,babilongv2_qa4_under_4k_instruct,babilongv2_qa5_under_4k_instruct

# hf (pretrained=unsloth/Llama-3.2-1B-Instruct), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 64
# |             Tasks              |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
# |--------------------------------|-------|------|-----:|------|---|-----:|---|------|
# |babilongv2_qa1_under_4k_instruct|    N/A|      |      |      |   |      |   |      |
# | - babilongv2_qa1_0k_instruct   |Yaml   |none  |     0|acc   |↑  |0.6870|±  |   N/A|
# | - babilongv2_qa1_1k_instruct   |Yaml   |none  |     0|acc   |↑  |0.6650|±  |   N/A|
# | - babilongv2_qa1_2k_instruct   |Yaml   |none  |     0|acc   |↑  |0.6080|±  |   N/A|
# | - babilongv2_qa1_4k_instruct   |Yaml   |none  |     0|acc   |↑  |0.5700|±  |   N/A|
# |babilongv2_qa2_under_4k_instruct|    N/A|      |      |      |   |      |   |      |
# | - babilongv2_qa2_0k_instruct   |Yaml   |none  |     0|acc   |↑  |0.4595|±  |   N/A|
# | - babilongv2_qa2_1k_instruct   |Yaml   |none  |     0|acc   |↑  |0.3473|±  |   N/A|
# | - babilongv2_qa2_2k_instruct   |Yaml   |none  |     0|acc   |↑  |0.2653|±  |   N/A|
# | - babilongv2_qa2_4k_instruct   |Yaml   |none  |     0|acc   |↑  |0.2142|±  |   N/A|
# |babilongv2_qa3_under_4k_instruct|    N/A|      |      |      |   |      |   |      |
# | - babilongv2_qa3_0k_instruct   |Yaml   |none  |     0|acc   |↑  |0.3443|±  |   N/A|
# | - babilongv2_qa3_1k_instruct   |Yaml   |none  |     0|acc   |↑  |0.3561|±  |   N/A|
# | - babilongv2_qa3_2k_instruct   |Yaml   |none  |     0|acc   |↑  |0.3106|±  |   N/A|
# | - babilongv2_qa3_4k_instruct   |Yaml   |none  |     0|acc   |↑  |0.2933|±  |   N/A|
# |babilongv2_qa4_under_4k_instruct|    N/A|      |      |      |   |      |   |      |
# | - babilongv2_qa4_0k_instruct   |Yaml   |none  |     0|acc   |↑  |0.1451|±  |   N/A|
# | - babilongv2_qa4_1k_instruct   |Yaml   |none  |     0|acc   |↑  |0.2392|±  |   N/A|
# | - babilongv2_qa4_2k_instruct   |Yaml   |none  |     0|acc   |↑  |0.3003|±  |   N/A|
# | - babilongv2_qa4_4k_instruct   |Yaml   |none  |     0|acc   |↑  |0.4444|±  |   N/A|
# |babilongv2_qa5_under_4k_instruct|    N/A|      |      |      |   |      |   |      |
# | - babilongv2_qa5_0k_instruct   |Yaml   |none  |     0|acc   |↑  |0.3373|±  |   N/A|
# | - babilongv2_qa5_1k_instruct   |Yaml   |none  |     0|acc   |↑  |0.5316|±  |   N/A|
# | - babilongv2_qa5_2k_instruct   |Yaml   |none  |     0|acc   |↑  |0.5335|±  |   N/A|
# | - babilongv2_qa5_4k_instruct   |Yaml   |none  |     0|acc   |↑  |0.6026|±  |   N/A|

# 03:30 FSDP
# {'babilongv2_qa1_under_4k_instruct': {' ': ' ', 'alias': 'babilongv2_qa1_under_4k_instruct'}, 'babilongv2_qa1_0k_instruct': {'alias': ' - babilongv2_qa1_0k_instruct', 'acc,none': 0.676, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa1_1k_instruct': {'alias': ' - babilongv2_qa1_1k_instruct', 'acc,none': 0.667, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa1_2k_instruct': {'alias': ' - babilongv2_qa1_2k_instruct', 'acc,none': 0.603, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa1_4k_instruct': {'alias': ' - babilongv2_qa1_4k_instruct', 'acc,none': 0.572, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa2_under_4k_instruct': {' ': ' ', 'alias': 'babilongv2_qa2_under_4k_instruct'}, 'babilongv2_qa2_0k_instruct': {'alias': ' - babilongv2_qa2_0k_instruct', 'acc,none': 0.45245245245245247, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa2_1k_instruct': {'alias': ' - babilongv2_qa2_1k_instruct', 'acc,none': 0.35135135135135137, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa2_2k_instruct': {'alias': ' - babilongv2_qa2_2k_instruct', 'acc,none': 0.2652652652652653, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa2_4k_instruct': {'alias': ' - babilongv2_qa2_4k_instruct', 'acc,none': 0.2032032032032032, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa3_under_4k_instruct': {' ': ' ', 'alias': 'babilongv2_qa3_under_4k_instruct'}, 'babilongv2_qa3_0k_instruct': {'alias': ' - babilongv2_qa3_0k_instruct', 'acc,none': 0.3433433433433433, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa3_1k_instruct': {'alias': ' - babilongv2_qa3_1k_instruct', 'acc,none': 0.36310904872389793, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa3_2k_instruct': {'alias': ' - babilongv2_qa3_2k_instruct', 'acc,none': 0.30561122244488975, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa3_4k_instruct': {'alias': ' - babilongv2_qa3_4k_instruct', 'acc,none': 0.2782782782782783, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa4_under_4k_instruct': {' ': ' ', 'alias': 'babilongv2_qa4_under_4k_instruct'}, 'babilongv2_qa4_0k_instruct': {'alias': ' - babilongv2_qa4_0k_instruct', 'acc,none': 0.01001001001001001, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa4_1k_instruct': {'alias': ' - babilongv2_qa4_1k_instruct', 'acc,none': 0.24724724724724725, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa4_2k_instruct': {'alias': ' - babilongv2_qa4_2k_instruct', 'acc,none': 0.2962962962962963, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa4_4k_instruct': {'alias': ' - babilongv2_qa4_4k_instruct', 'acc,none': 0.44744744744744747, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa5_under_4k_instruct': {' ': ' ', 'alias': 'babilongv2_qa5_under_4k_instruct'}, 'babilongv2_qa5_0k_instruct': {'alias': ' - babilongv2_qa5_0k_instruct', 'acc,none': 0.30430430430430433, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa5_1k_instruct': {'alias': ' - babilongv2_qa5_1k_instruct', 'acc,none': 0.5145436308926781, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa5_2k_instruct': {'alias': ' - babilongv2_qa5_2k_instruct', 'acc,none': 0.5315315315315315, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa5_4k_instruct': {'alias': ' - babilongv2_qa5_4k_instruct', 'acc,none': 0.5835835835835835, 'acc_stderr,none': 'N/A'}}

# 03:27 FSDP
# {'babilongv2_qa1_under_4k_instruct': {' ': ' ', 'alias': 'babilongv2_qa1_under_4k_instruct'}, 'babilongv2_qa1_0k_instruct': {'alias': ' - babilongv2_qa1_0k_instruct', 'acc,none': 0.68, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa1_1k_instruct': {'alias': ' - babilongv2_qa1_1k_instruct', 'acc,none': 0.668, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa1_2k_instruct': {'alias': ' - babilongv2_qa1_2k_instruct', 'acc,none': 0.608, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa1_4k_instruct': {'alias': ' - babilongv2_qa1_4k_instruct', 'acc,none': 0.567, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa2_under_4k_instruct': {' ': ' ', 'alias': 'babilongv2_qa2_under_4k_instruct'}, 'babilongv2_qa2_0k_instruct': {'alias': ' - babilongv2_qa2_0k_instruct', 'acc,none': 0.45245245245245247, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa2_1k_instruct': {'alias': ' - babilongv2_qa2_1k_instruct', 'acc,none': 0.35435435435435436, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa2_2k_instruct': {'alias': ' - babilongv2_qa2_2k_instruct', 'acc,none': 0.2652652652652653, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa2_4k_instruct': {'alias': ' - babilongv2_qa2_4k_instruct', 'acc,none': 0.2032032032032032, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa3_under_4k_instruct': {' ': ' ', 'alias': 'babilongv2_qa3_under_4k_instruct'}, 'babilongv2_qa3_0k_instruct': {'alias': ' - babilongv2_qa3_0k_instruct', 'acc,none': 0.34634634634634637, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa3_1k_instruct': {'alias': ' - babilongv2_qa3_1k_instruct', 'acc,none': 0.35730858468677495, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa3_2k_instruct': {'alias': ' - babilongv2_qa3_2k_instruct', 'acc,none': 0.3026052104208417, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa3_4k_instruct': {'alias': ' - babilongv2_qa3_4k_instruct', 'acc,none': 0.2752752752752753, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa4_under_4k_instruct': {' ': ' ', 'alias': 'babilongv2_qa4_under_4k_instruct'}, 'babilongv2_qa4_0k_instruct': {'alias': ' - babilongv2_qa4_0k_instruct', 'acc,none': 0.011011011011011011, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa4_1k_instruct': {'alias': ' - babilongv2_qa4_1k_instruct', 'acc,none': 0.24724724724724725, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa4_2k_instruct': {'alias': ' - babilongv2_qa4_2k_instruct', 'acc,none': 0.2962962962962963, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa4_4k_instruct': {'alias': ' - babilongv2_qa4_4k_instruct', 'acc,none': 0.44844844844844844, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa5_under_4k_instruct': {' ': ' ', 'alias': 'babilongv2_qa5_under_4k_instruct'}, 'babilongv2_qa5_0k_instruct': {'alias': ' - babilongv2_qa5_0k_instruct', 'acc,none': 0.3033033033033033, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa5_1k_instruct': {'alias': ' - babilongv2_qa5_1k_instruct', 'acc,none': 0.5155466399197592, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa5_2k_instruct': {'alias': ' - babilongv2_qa5_2k_instruct', 'acc,none': 0.5315315315315315, 'acc_stderr,none': 'N/A'}, 'babilongv2_qa5_4k_instruct': {'alias': ' - babilongv2_qa5_4k_instruct', 'acc,none': 0.5835835835835835, 'acc_stderr,none': 'N/A'}}



# hf (pretrained=unsloth/Llama-3.2-1B-Instruct), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 64
# |   Tasks    |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
# |------------|------:|------|-----:|------|---|-----:|---|-----:|
# |babilong_qa1|      0|none  |     2|acc   |↑  |0.7490|±  |0.0137|
# |babilong_qa2|      0|none  |     2|acc   |↑  |0.4454|±  |0.0157|
# |babilong_qa3|      0|none  |     2|acc   |↑  |0.3253|±  |0.0148|
# |babilong_qa4|      0|none  |     2|acc   |↑  |0.2282|±  |0.0133|
# |babilong_qa5|      0|none  |     2|acc   |↑  |0.2543|±  |0.0138|

# hf (pretrained=Qwen/Qwen2.5-7B-Instruct), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 32
# |   Tasks    |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
# |------------|------:|------|-----:|------|---|-----:|---|-----:|
# |babilong_qa1|      0|none  |     2|acc   |↑  |0.9840|±  |0.0040|
# |babilong_qa2|      0|none  |     2|acc   |↑  |0.6286|±  |0.0153|
# |babilong_qa3|      0|none  |     2|acc   |↑  |0.3794|±  |0.0154|
# |babilong_qa4|      0|none  |     2|acc   |↑  |0.5165|±  |0.0158|
# |babilong_qa5|      0|none  |     2|acc   |↑  |0.8599|±  |0.0110|


# 14 min 53 sec
# hf (pretrained=Qwen/Qwen2.5-7B-Instruct), gen_kwargs: (None), limit: None, num_fewshot: None, batch_size: 16
# |             Tasks              |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
# |--------------------------------|-------|------|-----:|------|---|-----:|---|------|
# |babilongv2_qa1_under_4k_instruct|    N/A|      |      |      |   |      |   |      |
# | - babilongv2_qa1_0k_instruct   |Yaml   |none  |     0|acc   |↑  |0.9840|±  |   N/A|
# | - babilongv2_qa1_1k_instruct   |Yaml   |none  |     0|acc   |↑  |0.9270|±  |   N/A|
# | - babilongv2_qa1_2k_instruct   |Yaml   |none  |     0|acc   |↑  |0.9240|±  |   N/A|
# | - babilongv2_qa1_4k_instruct   |Yaml   |none  |     0|acc   |↑  |0.9140|±  |   N/A|
# |babilongv2_qa2_under_4k_instruct|    N/A|      |      |      |   |      |   |      |
# | - babilongv2_qa2_0k_instruct   |Yaml   |none  |     0|acc   |↑  |0.6186|±  |   N/A|
# | - babilongv2_qa2_1k_instruct   |Yaml   |none  |     0|acc   |↑  |0.5415|±  |   N/A|
# | - babilongv2_qa2_2k_instruct   |Yaml   |none  |     0|acc   |↑  |0.5475|±  |   N/A|
# | - babilongv2_qa2_4k_instruct   |Yaml   |none  |     0|acc   |↑  |0.5245|±  |   N/A|
# |babilongv2_qa3_under_4k_instruct|    N/A|      |      |      |   |      |   |      |
# | - babilongv2_qa3_0k_instruct   |Yaml   |none  |     0|acc   |↑  |0.3614|±  |   N/A|
# | - babilongv2_qa3_1k_instruct   |Yaml   |none  |     0|acc   |↑  |0.4095|±  |   N/A|
# | - babilongv2_qa3_2k_instruct   |Yaml   |none  |     0|acc   |↑  |0.3828|±  |   N/A|
# | - babilongv2_qa3_4k_instruct   |Yaml   |none  |     0|acc   |↑  |0.3373|±  |   N/A|
# |babilongv2_qa4_under_4k_instruct|    N/A|      |      |      |   |      |   |      |
# | - babilongv2_qa4_0k_instruct   |Yaml   |none  |     0|acc   |↑  |0.5816|±  |   N/A|
# | - babilongv2_qa4_1k_instruct   |Yaml   |none  |     0|acc   |↑  |0.6346|±  |   N/A|
# | - babilongv2_qa4_2k_instruct   |Yaml   |none  |     0|acc   |↑  |0.6517|±  |   N/A|
# | - babilongv2_qa4_4k_instruct   |Yaml   |none  |     0|acc   |↑  |0.6436|±  |   N/A|
# |babilongv2_qa5_under_4k_instruct|    N/A|      |      |      |   |      |   |      |
# | - babilongv2_qa5_0k_instruct   |Yaml   |none  |     0|acc   |↑  |0.8809|±  |   N/A|
# | - babilongv2_qa5_1k_instruct   |Yaml   |none  |     0|acc   |↑  |0.8766|±  |   N/A|
# | - babilongv2_qa5_2k_instruct   |Yaml   |none  |     0|acc   |↑  |0.8749|±  |   N/A|
# | - babilongv2_qa5_4k_instruct   |Yaml   |none  |     0|acc   |↑  |0.8639|±  |   N/A|