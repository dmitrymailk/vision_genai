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
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    # print(accelerator.unwrap_model(model).config)
    eval_model = SimpleAccelerateHFLM(
        pretrained=model,
        accelerator=accelerator,
        tokenizer=tokenizer,
        config=config,
    )

    results = evaluator.simple_evaluate(
        # model=HFLM(pretrained=model, tokenizer=tokenizer),
        model=eval_model,
        tasks=[
            "arc_easy",
            "hellaswag",
        ],
        verbosity="WARNING",
        batch_size=64,
    )
    if eval_model._rank == 0:
        print(results["results"])


# {'arc_easy': {'alias': 'arc_easy', 'acc,none': 0.6548821548821548, 'acc_stderr,none': 0.009755139387152048, 'acc_norm,none': 0.6052188552188552, 'acc_norm_stderr,none': 0.01003003893588358}, 'hellaswag': {'alias': 'hellaswag', 'acc,none': 0.477096195976897, 'acc_stderr,none': 0.004984543540932336, 'acc_norm,none': 0.6363274248157738, 'acc_norm_stderr,none': 0.004800728138792352}}
