import warnings

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from train_gym.massive_train.evaluation.custom_lm_eval import SimpleHFLM
from train_gym.massive_train.evaluation.custom_lm_eval_v2 import SimpleAccelerateHFLM
from accelerate import Accelerator

if __name__ == "__main__":
    # model_name = "unsloth/Llama-3.2-1B-Instruct"
    accelerator = Accelerator()

    with accelerator.main_process_first():
        model_name = "unsloth/Llama-3.2-1B"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            # device_map={"": 0},
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
    # eval_model = HFLM(
    #     pretrained=model_name,
    # )
    # eval_model = SimpleAccelerateHFLM(
    #     pretrained=model_name,
    # )
    use_fsdp = True
    # use_fsdp = False
    if use_fsdp:
        no_decay = ["bias", "layer_norm.weight"]

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
        )
        # model = accelerator.prepare(model)
        model, optimizer = accelerator.prepare(model, optimizer)
        # optimizer = accelerator.prepare(optimizer)
    else:
        accelerator = Accelerator()
        model = accelerator.prepare(model)

    # print(accelerator.unwrap_model(model).config)
    eval_model = SimpleAccelerateHFLM(
        pretrained=model,
        accelerator=accelerator,
        tokenizer=tokenizer,
        config=config,
        batch_size=64,
        # batch_size=32,
        # mixed_precision_dtype=torch.bfloat16,
        # mixed_precision_dtype="bf16",
    )

    results = evaluator.simple_evaluate(
        # model=HFLM(pretrained=model, tokenizer=tokenizer),
        model=eval_model,
        tasks=[
            "arc_easy",
            "hellaswag",
            "global_mmlu_en_stem",
            # "squadv2",
        ],
        verbosity="WARNING",
        batch_size=64,
        # limit=300,
    )

    if eval_model._rank == 0:
        print(results["results"])


# {'arc_easy': {'alias': 'arc_easy', 'acc,none': 0.6548821548821548, 'acc_stderr,none': 0.009755139387152048, 'acc_norm,none': 0.6052188552188552, 'acc_norm_stderr,none': 0.01003003893588358}, 'hellaswag': {'alias': 'hellaswag', 'acc,none': 0.477096195976897, 'acc_stderr,none': 0.004984543540932336, 'acc_norm,none': 0.6363274248157738, 'acc_norm_stderr,none': 0.004800728138792352}}

# fsdp2 A100-80GB 4GPU, arc_easy, hellaswag - 7 min 27 sec


# FSDP, 4GPU batch 32, squadv2, больше 55 min 20 sec, 8.75s/it
# DDP, 4GPU batch 32, squadv2, больше 10 min 20 sec, 1.15s/it
