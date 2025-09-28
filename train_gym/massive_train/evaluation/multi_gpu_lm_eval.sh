export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_VISIBLE_DEVICES=2,3

model_name=unsloth/Llama-3.2-1B
accelerate launch --mixed_precision bf16 -m lm_eval \
    --model hf \
    --model_args pretrained=$model_name \
    --tasks arc_easy,hellaswag,winogrande,sciq,copa,openbookqa,mmlu_stem,mmlu_other,mmlu_social_sciences,mmlu_humanities \
    --batch_size 32
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