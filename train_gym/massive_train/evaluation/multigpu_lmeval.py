import warnings

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from train_gym.massive_train.evaluation.custom_lm_eval import SimpleHFLM
from train_gym.massive_train.evaluation.custom_lm_eval_v2 import SimpleAccelerateHFLM
from accelerate import Accelerator
from train_gym.rmt.rmt_wrappers import (
    MemoryCell,
    RecurrentWrapper,
)
from train_gym.rmt.hf_rmt_wrappers import (
    RMTForReasoning,
    RMTConfig,
)


if __name__ == "__main__":
    # model_name = "unsloth/Llama-3.2-1B-Instruct"
    accelerator = Accelerator()

    with accelerator.main_process_first():
        # model_name = "unsloth/Llama-3.2-1B"
        # model_name = "HuggingFaceTB/SmolLM2-360M"
        model_name = "unsloth/Llama-3.2-1B-Instruct"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            # device_map={"": 0},
        )
        config = AutoConfig.from_pretrained(model_name)
        # model = RMTForReasoning.from_pretrained(
        #     model_name,
        #     dtype=torch.bfloat16,
        #     attn_implementation="flash_attention_2",
        # )
        cell = MemoryCell(
            model,
            num_mem_tokens=32,
        )
        model = RecurrentWrapper(
            cell,
            segment_size=1024,
            max_n_segments=2,
            vary_n_segments=False,
            k2=-1,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # config = AutoConfig.from_pretrained(model_name)

        config.use_cache = True

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
        model, optimizer = accelerator.prepare(model, optimizer)

    else:
        accelerator = Accelerator()
        model = accelerator.prepare(model)

    # batch_size = 1
    # batch_size = 64
    # batch_size = 56
    batch_size = 32
    # batch_size = 8
    # batch_size = 16
    eval_model = SimpleAccelerateHFLM(
        pretrained=model,
        accelerator=accelerator,
        tokenizer=tokenizer,
        config=config,
        # batch_size=64,
        batch_size=batch_size,
        # batch_size=32,
        # mixed_precision_dtype=torch.bfloat16,
        # mixed_precision_dtype="bf16",
    )

    results = evaluator.simple_evaluate(
        # model=HFLM(pretrained=model, tokenizer=tokenizer),
        model=eval_model,
        tasks=[
            # "arc_easy",
            # "hellaswag",
            # "winogrande",
            # "sciq",
            # "copa",
            # "openbookqa",
            # "mmlu_stem",
            # "mmlu_other",
            # "mmlu_social_sciences",
            # "mmlu_humanities",
            # "babilongv2_qa1_under_4k_instruct",
            # "babilongv2_qa2_under_4k_instruct",
            # "babilongv2_qa3_under_4k_instruct",
            # "babilongv2_qa4_under_4k_instruct",
            # "babilongv2_qa5_under_4k_instruct",
            # "babilongv2_qa1_under_4k_base",
            # "babilongv2_qa2_under_4k_base",
            # "babilongv2_qa3_under_4k_base",
            # "babilongv2_qa4_under_4k_base",
            # "babilongv2_qa5_under_4k_base",
            # "babilongv2_qa1_0k_instruct",
            # "babilongv2_qa1_0k_base",
            # "arc_easy",
            "babilongv2_qa1_0k_base",
            "babilongv2_qa1_1k_base",
            "babilongv2_qa1_2k_base",
            "babilongv2_qa1_4k_base",
        ],
        verbosity="WARNING",
        # batch_size=64,
        batch_size=batch_size,
        # limit=300,
        # apply_chat_template=True,
    )

    if eval_model._rank == 0:
        print(results["results"])


# {'arc_easy': {'alias': 'arc_easy', 'acc,none': 0.6548821548821548, 'acc_stderr,none': 0.009755139387152048, 'acc_norm,none': 0.6052188552188552, 'acc_norm_stderr,none': 0.01003003893588358}, 'hellaswag': {'alias': 'hellaswag', 'acc,none': 0.477096195976897, 'acc_stderr,none': 0.004984543540932336, 'acc_norm,none': 0.6363274248157738, 'acc_norm_stderr,none': 0.004800728138792352}}
