
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