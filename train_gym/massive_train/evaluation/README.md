
git clone https://github.com/EleutherAI/lm-evaluation-harness.git

cd lm-evaluation-harness && touch lm_eval/models/custom_model_1.py

## parallel evaluation lm eval

### Data paralell v1

```python
import warnings

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM


if __name__ == "__main__":

    model_name = "unsloth/Llama-3.2-1B"
    eval_model = HFLM(
        pretrained=model_name,
    )
    results = evaluator.simple_evaluate(
        model=eval_model,
        tasks=[
            "arc_easy",
            "hellaswag",
        ],
        verbosity="WARNING",
        batch_size=32,
    )

    if eval_model._rank == 0:
        print(results["results"])
```
```yaml
distributed_type: "MULTI_GPU"
# Can be one of "no", "fp16", or "bf16" (see `transformer_engine.yaml` for `fp8`)
mixed_precision: "bf16"
# Specify the number of GPUs to use
num_processes: 2
```
```bash
export CUDA_VISIBLE_DEVICES=0,1

# config_path=/code/train_gym/massive_train/evaluation/fsdp2_default_config.yaml
config_path=/code/train_gym/massive_train/evaluation/multi_gpu.yaml

accelerate launch --config_file $config_path multigpu_lmeval.py
```

### Data paralell v2, passing accelerate
```python
import warnings

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from train_gym.massive_train.evaluation.custom_lm_eval_v2 import SimpleAccelerateHFLM
from accelerate import Accelerator

if __name__ == "__main__":

    
    model_name = "unsloth/Llama-3.2-1B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    
    eval_model = SimpleAccelerateHFLM(
        pretrained=model,
        accelerator=accelerator,
        tokenizer=tokenizer,
        config=config,
    )

    results = evaluator.simple_evaluate(
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
```
```yaml
distributed_type: "MULTI_GPU"
# Can be one of "no", "fp16", or "bf16" (see `transformer_engine.yaml` for `fp8`)
mixed_precision: "bf16"
# Specify the number of GPUs to use
num_processes: 2
```
```bash
export CUDA_VISIBLE_DEVICES=0,1

# config_path=/code/train_gym/massive_train/evaluation/fsdp2_default_config.yaml
config_path=/code/train_gym/massive_train/evaluation/multi_gpu.yaml

accelerate launch --config_file $config_path multigpu_lmeval.py
```

### FSDP

```python
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
            # "arc_easy",
            # "hellaswag",
            "global_mmlu_en_stem",
        ],
        verbosity="WARNING",
        batch_size=64,
        limit=300,
    )

    if eval_model._rank == 0:
        print(results["results"])
```
```yaml
compute_environment: LOCAL_MACHINE                                                                                                             
debug: false                                                                                                                                   
distributed_type: FSDP
downcast_bf16: 'no'
enable_cpu_affinity: false
fsdp_config:
  fsdp_activation_checkpointing: false
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_cpu_ram_efficient_loading: true
  fsdp_offload_params: false
  fsdp_reshard_after_forward: true
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
  fsdp_version: 2
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

```bash
pushd /code/

export CUDA_VISIBLE_DEVICES="0,1,2,3"
config_path=/code/train_gym/massive_train/evaluation/fsdp2_default_config.yaml

accelerate launch --num_processes=4 --config_file $config_path -m train_gym.massive_train.evaluation.multigpu_lmeval
```