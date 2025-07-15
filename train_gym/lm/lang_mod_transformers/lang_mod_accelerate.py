#!/usr/bin/env python
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
import logging
import math
import os

os.environ["WANDB_PROJECT"] = "llm_pretraining_optimization"
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import evaluate
import torch
from datasets import load_dataset
import gc
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_xla_available,
    set_seed,
    get_scheduler,
)

# from torchtune.models.llama3_2 import llama3_2_1b
from transformers.testing_utils import CaptureLogger
from transformers.utils.versions import require_version
from liger_kernel.transformers.functional import liger_cross_entropy
from typing import Any, Sequence, cast
from cut_cross_entropy.transformers import cce_patch
from transformers import DataCollatorWithFlattening
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
)

from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.loss.loss_utils import nn
from functools import partial
from accelerate.utils import compile_regions
from lang_mod_transformers.utils import (
    ModelArguments,
    DataTrainingArguments,
    cuda_streams_forward,
)
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from types import MethodType
from torchao.float8 import convert_to_float8_training, Float8LinearConfig
from accelerate.utils import FP8RecipeKwargs, TERecipeKwargs

# from transformers.trainer_pt_utils import AcceleratorConfig
from accelerate import Accelerator, DistributedType

from tqdm import tqdm
from torch.utils.data import (
    DataLoader,
    Dataset,
    IterableDataset,
    RandomSampler,
    SequentialSampler,
)


from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    SaveStrategy,
    TrainerMemoryTracker,
    TrainOutput,
    check_target_module_exists,
    default_compute_objective,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    neftune_post_forward_hook,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)
from accelerate.utils import DataLoaderConfiguration
from accelerate.utils.transformer_engine import convert_model
from lang_mod_transformers.llama3_2_hf_v2 import (
    LlamaForCausalLM as LlamaForCausalLMHF_V2,
)
from lang_mod_transformers import llama3_2_hf_v2
from cut_cross_entropy.transformers.llama import (
    cce_forward,
    linear_cross_entropy,
    _PATCH_OPTS,
)
from cut_cross_entropy.transformers.utils import PatchOptions
from lang_mod_transformers.llama3_2_torchtune_v3 import (
    llama3_2_1b,
    hf_to_tune,
    TransformerDecoder,
    TransformerSelfAttentionLayer,
)

# from
# from transformer_engine.common.recipe import DelayedScaling

# torch.backends.cudnn.allow_tf32 = True
# torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def filter_linear_layers(module, fqn, first_layer_name=None, last_layer_name=None):
    if isinstance(module, torch.nn.Linear):
        if module.in_features % 16 != 0 or module.out_features % 16 != 0:
            return False
    # For stability reasons, we skip the first and last linear layers
    # Otherwise can lead to the model not training or converging properly
    if fqn in (first_layer_name, last_layer_name):
        return False
    return True


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
    )
    model_name_or_path = model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        use_fast=True,
    )

    torch_dtype = torch.bfloat16
    data_collator = default_data_collator
    optimization_level = model_args.optimization_level
    original_forward = LlamaForCausalLM.forward
    config = AutoConfig.from_pretrained(
        model_name_or_path,
    )
    accelerator_log_kwargs = {
        "log_with": "wandb",
        "project_dir": "train_output",
        # Убираем "mixed_precision": "fp8" чтобы избежать конфликта с bf16
    }
    accelerator = None
    match optimization_level:
        case "opt_1":
            print("opt_1")
            # https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.from_pretrained.attn_implementation
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                attn_implementation=model_args.attn_implementation,
                # device_map={"": 0},
            )
            # model = AutoModelForCausalLM.from_config(
            #     config,
            #     torch_dtype=torch_dtype,
            #     attn_implementation=model_args.attn_implementation,
            # )
            dataloader_params = [
                "split_batches",
                "dispatch_batches",
                "even_batches",
                "use_seedable_sampler",
            ]

            dataloader_config = DataLoaderConfiguration(
                **{
                    param: training_args.accelerator_config.pop(param)
                    for param in dataloader_params
                }
            )
            accelerator = Accelerator(
                mixed_precision="bf16",
                dataloader_config=dataloader_config,
                gradient_accumulation_steps=training_args.gradient_accumulation_steps,
                **accelerator_log_kwargs,
            )
        case "opt_25":
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                attn_implementation=model_args.attn_implementation,
            )
            # model = AutoModelForCausalLM.from_config(
            #     config,
            #     torch_dtype=torch_dtype,
            #     attn_implementation=model_args.attn_implementation,
            # )
            # unsloth
            cross_entropy_impl = "cce"
            model = cce_patch(
                model,
                cross_entropy_impl,
            )
            FP8_RECIPE_KWARGS = {
                "fp8_format": "HYBRID",
                # "fp8_format": "E4M3",
                "amax_history_len": 32,
                # "amax_history_len": None,
                # "amax_compute_algo": "most_recent",
                "amax_compute_algo": "max",
                "backend": "TE",
            }
            # FP8_RECIPE_KWARGS = {
            #     "opt_level": "O2",
            #     "backend": "msamp",
            # }
            kwargs_handlers = [
                FP8RecipeKwargs(
                    **FP8_RECIPE_KWARGS,
                )
            ]
            # AcceleratorState()._reset_state(True)
            accelerator = Accelerator(
                mixed_precision="fp8",
                kwargs_handlers=kwargs_handlers,
                **accelerator_log_kwargs,
            )
            # accelerator = Accelerator(
            #     **accelerator_log_kwargs,
            # )
        case "opt_26":
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                attn_implementation=model_args.attn_implementation,
            )
            # model = AutoModelForCausalLM.from_config(
            #     config,
            #     torch_dtype=torch_dtype,
            #     attn_implementation=model_args.attn_implementation,
            # )
            # unsloth
            cross_entropy_impl = "cce"
            model = cce_patch(
                model,
                cross_entropy_impl,
            )
            FP8_RECIPE_KWARGS = {
                "fp8_format": "HYBRID",
                # "fp8_format": "E4M3",
                "amax_history_len": 32,
                # "amax_history_len": None,
                # "amax_compute_algo": "most_recent",
                "amax_compute_algo": "max",
                "backend": "TE",
            }
            # FP8_RECIPE_KWARGS = {
            #     "opt_level": "O2",
            #     "backend": "msamp",
            # }
            kwargs_handlers = [
                FP8RecipeKwargs(
                    **FP8_RECIPE_KWARGS,
                )
            ]
            # AcceleratorState()._reset_state(True)
            accelerator = Accelerator(
                mixed_precision="fp8",
                kwargs_handlers=kwargs_handlers,
                **accelerator_log_kwargs,
            )
            # accelerator = Accelerator(
            #     **accelerator_log_kwargs,
            # )
            for m in reversed(list(model.modules())):
                if isinstance(m, LlamaDecoderLayer):
                    m.compile(
                        backend="inductor",
                        # mode="max-autotune",
                    )
        case "opt_27":
            # Возвращаем объект с loss как в Transformers
            from dataclasses import dataclass
            from typing import Optional, Tuple

            @dataclass
            class CausalLMOutputWithLoss:
                loss: Optional[torch.FloatTensor] = None
                logits: torch.FloatTensor = None
                hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
                attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

            print("opt_27")
            # Загружаем модель Llama 3.2 из TorchTune
            # Доступна только llama3_2_1b с предустановленными параметрами
            torchtune_model = llama3_2_1b(
                tie_word_embeddings=True,
            )

            # Загружаем веса из оригинальной HuggingFace модели
            print(f"Loading weights from {model_name_or_path} into TorchTune model")
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="cpu",  # Загружаем на CPU для копирования весов
            )

            # Используем готовую функцию hf_to_tune для преобразования весов
            # from torchtune.models.convert_weights import hf_to_tune

            # Получаем state_dict из HF модели
            hf_state_dict = hf_model.state_dict()

            # Параметры для Llama 3.2 1B
            # Получаем параметры из конфигурации HF модели
            hf_config = hf_model.config
            num_heads = hf_config.num_attention_heads
            num_kv_heads = getattr(
                hf_config, "num_key_value_heads", num_heads
            )  # GQA может не быть
            dim = hf_config.hidden_size
            head_dim = hf_config.hidden_size // hf_config.num_attention_heads

            print(
                f"Model config: num_heads={num_heads}, num_kv_heads={num_kv_heads}, dim={dim}, head_dim={head_dim}"
            )

            # Преобразуем веса из HF формата в TorchTune формат
            converted_state_dict = hf_to_tune(
                hf_state_dict,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                dim=dim,
                head_dim=head_dim,
            )

            # Загружаем преобразованные веса в TorchTune модель
            torchtune_state_dict = torchtune_model.state_dict()

            # Применяем преобразованные веса к модели
            copied_count = 0
            for key, value in converted_state_dict.items():
                if key in torchtune_state_dict:
                    if value.shape == torchtune_state_dict[key].shape:
                        torchtune_state_dict[key].copy_(value)
                        copied_count += 1
                    else:
                        print(
                            f"Shape mismatch for {key}: converted {value.shape} vs TorchTune {torchtune_state_dict[key].shape}"
                        )
                else:
                    print(f"Parameter {key} not found in TorchTune model")

            print(f"Copied {copied_count} parameters from HF model to TorchTune model")

            # Загружаем обновленный state_dict в модель
            torchtune_model.load_state_dict(torchtune_state_dict)

            # Очищаем память
            del hf_model
            torch.cuda.empty_cache()

            gc.collect()

            # Инициализируем модель в bfloat16
            torchtune_model = torchtune_model.to(dtype=torch.bfloat16)

            # Создаем обертку для совместимости с Transformers
            class TorchTuneWrapper(torch.nn.Module):
                def __init__(self, model: TransformerDecoder):
                    super().__init__()
                    self.model = model

                def forward(
                    self, input_ids=None, attention_mask=None, labels=None, **kwargs
                ):
                    # TorchTune модель ожидает input_ids и mask
                    # Для простоты используем None - TorchTune автоматически создаст causal mask
                    # Это должно работать с torch.compile
                    # outputs = self.model(input_ids, mask=attention_mask)

                    # Если есть labels, вычисляем loss как в Transformers
                    if labels is not None:
                        # Shift logits and labels для causal LM
                        shift_logits = outputs[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()

                        # Используем cut-cross-entropy loss
                        # loss = torch.nn.functional.cross_entropy(
                        loss = liger_cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            ignore_index=-100,
                        )

                        return CausalLMOutputWithLoss(loss=loss, logits=outputs)

                    return outputs

            model = TorchTuneWrapper(torchtune_model)

            # Применяем torch.compile к отдельным декодер блокам
            # for m in reversed(list(model.model.modules())):
            #     if (
            #         isinstance(m, torch.nn.Module)
            #         and hasattr(m, "attn")
            #         and hasattr(m, "mlp")
            #     ):
            #         # Это TransformerSelfAttentionLayer
            #         m.compile(
            #             backend="inductor",
            #             # работает только на torch 2.8.0 и выше, меньше выдает ошибку
            #             # прирост минимален, сотые доли, но есть
            #             # mode="max-autotune",
            #         )
            # работает слегка хуже, чем копиляция каждого блока отдельно
            # model.model = torch.compile(
            #     model.model,
            #     backend="inductor",
            #     # mode="max-autotune",
            # )

            # Настройка accelerator для TorchTune модели
            accelerator = Accelerator(
                mixed_precision="bf16",
                **accelerator_log_kwargs,
            )
        case "opt_28":
            print("opt_28")
            model = LlamaForCausalLMHF_V2.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                use_cache=False,
                torch_dtype=torch_dtype,
            )
            # model = LlamaForCausalLM.from_pretrained(
            #     model_name_or_path,
            #     trust_remote_code=True,
            #     use_cache=False,
            #     torch_dtype=torch_dtype,
            # )
            # unsloth
            cross_entropy_impl = "cce"
            # model = cce_patch(
            #     model,
            #     cross_entropy_impl,
            # )
            model.forward = MethodType(cce_forward, model)
            first_linear = None
            last_linear = None
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    if first_linear is None:
                        first_linear = name
                    last_linear = name

            func = partial(
                filter_linear_layers,
                first_layer_name=first_linear,
                last_layer_name=last_linear,
            )
            config = Float8LinearConfig.from_recipe_name("tensorwise")
            convert_to_float8_training(
                model,
                config=config,
                module_filter_fn=func,
            )
            # model = torch.compile(model)
            for m in reversed(list(model.modules())):
                if isinstance(m, llama3_2_hf_v2.LlamaDecoderLayer):
                    m.compile(
                        backend="inductor",
                        # mode="max-autotune",
                    )
            # for m in reversed(list(model.modules())):
            #     if isinstance(m, LlamaDecoderLayer):
            #         m.compile(
            #             backend="inductor",
            #             mode="max-autotune",
            #         )

            dataloader_params = [
                "split_batches",
                "dispatch_batches",
                "even_batches",
                "use_seedable_sampler",
            ]

            dataloader_config = DataLoaderConfiguration(
                **{
                    param: training_args.accelerator_config.pop(param)
                    for param in dataloader_params
                }
            )
            accelerator = Accelerator(
                mixed_precision="bf16",
                dataloader_config=dataloader_config,
                gradient_accumulation_steps=training_args.gradient_accumulation_steps,
                **accelerator_log_kwargs,
            )
        case "opt_29":
            # Возвращаем объект с loss как в Transformers
            from dataclasses import dataclass
            from typing import Optional, Tuple

            @dataclass
            class CausalLMOutputWithLoss:
                loss: Optional[torch.FloatTensor] = None
                logits: torch.FloatTensor = None
                hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
                attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

            print("opt_29")
            # Загружаем модель Llama 3.2 из TorchTune
            # Доступна только llama3_2_1b с предустановленными параметрами
            torchtune_model = llama3_2_1b(
                tie_word_embeddings=True,
            )

            # Загружаем веса из оригинальной HuggingFace модели
            print(f"Loading weights from {model_name_or_path} into TorchTune model")
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="cpu",  # Загружаем на CPU для копирования весов
            )

            # Используем готовую функцию hf_to_tune для преобразования весов
            # from torchtune.models.convert_weights import hf_to_tune

            # Получаем state_dict из HF модели
            hf_state_dict = hf_model.state_dict()

            # Параметры для Llama 3.2 1B
            # Получаем параметры из конфигурации HF модели
            hf_config = hf_model.config
            num_heads = hf_config.num_attention_heads
            num_kv_heads = getattr(
                hf_config, "num_key_value_heads", num_heads
            )  # GQA может не быть
            dim = hf_config.hidden_size
            head_dim = hf_config.hidden_size // hf_config.num_attention_heads

            print(
                f"Model config: num_heads={num_heads}, num_kv_heads={num_kv_heads}, dim={dim}, head_dim={head_dim}"
            )

            # Преобразуем веса из HF формата в TorchTune формат
            converted_state_dict = hf_to_tune(
                hf_state_dict,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                dim=dim,
                head_dim=head_dim,
            )

            # Загружаем преобразованные веса в TorchTune модель
            torchtune_state_dict = torchtune_model.state_dict()

            # Применяем преобразованные веса к модели
            for key, value in converted_state_dict.items():
                if key in torchtune_state_dict:
                    if value.shape == torchtune_state_dict[key].shape:
                        torchtune_state_dict[key].copy_(value)
                    else:
                        print(
                            f"Shape mismatch for {key}: converted {value.shape} vs TorchTune {torchtune_state_dict[key].shape}"
                        )
                else:
                    print(f"Parameter {key} not found in TorchTune model")

            # Загружаем обновленный state_dict в модель
            torchtune_model.load_state_dict(torchtune_state_dict)

            # Очищаем память
            del hf_model
            gc.collect()
            torch.cuda.empty_cache()

            # Инициализируем модель в bfloat16
            torchtune_model = torchtune_model.to(dtype=torch.bfloat16)

            # Применяем float8 оптимизацию как в opt_21-23
            first_linear = None
            last_linear = None
            for name, module in torchtune_model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    if first_linear is None:
                        first_linear = name
                    last_linear = name

            func = partial(
                filter_linear_layers,
                first_layer_name=first_linear,
                last_layer_name=last_linear,
            )
            config = Float8LinearConfig.from_recipe_name("tensorwise")
            convert_to_float8_training(
                torchtune_model,
                config=config,
                module_filter_fn=func,
            )

            # Создаем обертку для совместимости с Transformers
            class TorchTuneWrapper(torch.nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(
                    self, input_ids=None, attention_mask=None, labels=None, **kwargs
                ):
                    # TorchTune модель ожидает input_ids и mask
                    # Для простоты используем None - TorchTune автоматически создаст causal mask
                    # Это должно работать с torch.compile
                    outputs = self.model(input_ids, mask=None)

                    # Если есть labels, вычисляем loss как в Transformers
                    if labels is not None:
                        # Shift logits and labels для causal LM
                        shift_logits = outputs[..., :-1, :].contiguous()
                        shift_labels = labels[..., 1:].contiguous()

                        # Используем cut-cross-entropy loss
                        # loss = torch.nn.functional.cross_entropy(
                        loss = liger_cross_entropy(
                            shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1),
                            ignore_index=-100,
                        )

                        return CausalLMOutputWithLoss(loss=loss, logits=outputs)

                    return outputs

            model = TorchTuneWrapper(torchtune_model)

            # Применяем torch.compile к отдельным декодер блокам
            for m in reversed(list(model.model.modules())):
                if (
                    isinstance(m, torch.nn.Module)
                    and hasattr(m, "attn")
                    and hasattr(m, "mlp")
                ):
                    # Это TransformerSelfAttentionLayer
                    m.compile(
                        backend="inductor",
                        # работает только на torch 2.8.0 и выше, меньше выдает ошибку
                        # прирост минимален, сотые доли, но есть
                        # mode="max-autotune",
                    )

            # Настройка accelerator для TorchTune модели
            accelerator = Accelerator(
                mixed_precision="bf16",
                **accelerator_log_kwargs,
            )
        case "opt_30":
            # Возвращаем объект с loss как в Transformers
            from dataclasses import dataclass
            from typing import Optional, Tuple

            @dataclass
            class CausalLMOutputWithLoss:
                loss: Optional[torch.FloatTensor] = None
                logits: torch.FloatTensor = None
                hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
                attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

            print("opt_30")
            # Загружаем модель Llama 3.2 из TorchTune
            # Доступна только llama3_2_1b с предустановленными параметрами
            torchtune_model = llama3_2_1b(
                tie_word_embeddings=True,
            )

            # Загружаем веса из оригинальной HuggingFace модели
            print(f"Loading weights from {model_name_or_path} into TorchTune model")
            hf_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="cpu",  # Загружаем на CPU для копирования весов
            )

            # Используем готовую функцию hf_to_tune для преобразования весов
            # from torchtune.models.convert_weights import hf_to_tune

            # Получаем state_dict из HF модели
            hf_state_dict = hf_model.state_dict()

            # Параметры для Llama 3.2 1B
            # Получаем параметры из конфигурации HF модели
            hf_config = hf_model.config
            num_heads = hf_config.num_attention_heads
            num_kv_heads = getattr(
                hf_config, "num_key_value_heads", num_heads
            )  # GQA может не быть
            dim = hf_config.hidden_size
            head_dim = hf_config.hidden_size // hf_config.num_attention_heads

            print(
                f"Model config: num_heads={num_heads}, num_kv_heads={num_kv_heads}, dim={dim}, head_dim={head_dim}"
            )

            # Преобразуем веса из HF формата в TorchTune формат
            converted_state_dict = hf_to_tune(
                hf_state_dict,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                dim=dim,
                head_dim=head_dim,
            )

            # Загружаем преобразованные веса в TorchTune модель
            torchtune_state_dict = torchtune_model.state_dict()

            # Применяем преобразованные веса к модели
            for key, value in converted_state_dict.items():
                if key in torchtune_state_dict:
                    if value.shape == torchtune_state_dict[key].shape:
                        torchtune_state_dict[key].copy_(value)
                    else:
                        print(
                            f"Shape mismatch for {key}: converted {value.shape} vs TorchTune {torchtune_state_dict[key].shape}"
                        )
                else:
                    print(f"Parameter {key} not found in TorchTune model")

            # Загружаем обновленный state_dict в модель
            torchtune_model.load_state_dict(torchtune_state_dict)

            # Очищаем память
            del hf_model
            gc.collect()
            torch.cuda.empty_cache()

            # Инициализируем модель в bfloat16
            torchtune_model = torchtune_model.to(dtype=torch.bfloat16)

            # Применяем float8 оптимизацию как в opt_21-23
            first_linear = None
            last_linear = None
            for name, module in torchtune_model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    if first_linear is None:
                        first_linear = name
                    last_linear = name

            func = partial(
                filter_linear_layers,
                first_layer_name=first_linear,
                last_layer_name=last_linear,
            )
            config = Float8LinearConfig.from_recipe_name("tensorwise")
            convert_to_float8_training(
                torchtune_model,
                config=config,
                module_filter_fn=func,
            )

            # Создаем обертку для совместимости с Transformers
            class TorchTuneWrapper(torch.nn.Module):
                def __init__(self, model: TransformerDecoder):
                    super().__init__()
                    self.model = model

                def forward(
                    self, input_ids=None, attention_mask=None, labels=None, **kwargs
                ):
                    # TorchTune модель ожидает input_ids и mask
                    # Для простоты используем None - TorchTune автоматически создаст causal mask
                    # Это должно работать с torch.compile
                    # outputs = self.model(input_ids, mask=attention_mask)

                    # # Если есть labels, вычисляем loss как в Transformers
                    # if labels is not None:
                    #     # Shift logits and labels для causal LM
                    #     shift_logits = outputs[..., :-1, :].contiguous()
                    #     shift_labels = labels[..., 1:].contiguous()

                    #     # Используем cut-cross-entropy loss
                    #     # loss = torch.nn.functional.cross_entropy(
                    #     loss = liger_cross_entropy(
                    #         shift_logits.view(-1, shift_logits.size(-1)),
                    #         shift_labels.view(-1),
                    #         ignore_index=-100,
                    #     )

                    #     return CausalLMOutputWithLoss(loss=loss, logits=outputs)
                    tokens = input_ids
                    mask = None
                    encoder_input = None
                    encoder_mask = None
                    input_pos = None
                    input_embeds = None
                    # self.model._validate_inputs(
                    #     tokens=tokens,
                    #     mask=mask,
                    #     encoder_input=encoder_input,
                    #     encoder_mask=encoder_mask,
                    #     input_pos=input_pos,
                    #     input_embeds=input_embeds,
                    # )

                    # shape: [b, s, d]
                    h = self.model.tok_embeddings(tokens)

                    # hidden = []
                    for i, layer in enumerate(self.model.layers):
                        # if i in self.output_hidden_states:
                        #     hidden.append(h)
                        # shape: [b, s, d]
                        h = layer(
                            h,
                            mask=mask,
                            encoder_input=encoder_input,
                            encoder_mask=encoder_mask,
                            input_pos=input_pos,
                        )

                    h = self.model.norm(h)
                    # if len(self.layers) in self.output_hidden_states:
                    #     hidden.append(h)
                    loss = None
                    logits = None
                    if labels is not None and _PATCH_OPTS is not None:
                        loss = linear_cross_entropy(
                            h,
                            self.model.tok_embeddings.weight,
                            labels.to(h.device),
                            shift=True,
                            impl=_PATCH_OPTS.impl,
                            reduction=_PATCH_OPTS.reduction,
                        )
                        return CausalLMOutputWithLoss(loss=loss)
                    # shape: [b, seq_len, out_dim]
                    output = self.model.unembed(h)

                    return CausalLMOutputWithLoss(loss=loss, logits=output)

            model = TorchTuneWrapper(torchtune_model)

            # Применяем torch.compile к отдельным декодер блокам
            for m in reversed(list(model.model.modules())):
                if (
                    isinstance(m, torch.nn.Module)
                    and hasattr(m, "attn")
                    and hasattr(m, "mlp")
                ):
                    # Это TransformerSelfAttentionLayer
                    m.compile(
                        backend="inductor",
                        # работает только на torch 2.8.0 и выше, меньше выдает ошибку
                        # прирост минимален, сотые доли, но есть
                        # mode="max-autotune",
                    )

            # Настройка accelerator для TorchTune модели
            accelerator = Accelerator(
                mixed_precision="bf16",
                **accelerator_log_kwargs,
            )

    print("model_args.attn_implementation", model_args.attn_implementation)

    # Preprocessing the datasets.
    column_names = list(raw_datasets["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger(
        "transformers.tokenization_utils_base"
    )

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    # block_size = data_args.block_size
    block_size = 1024

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]
    # print(train_dataset[0])

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)

    print(training_args)
    # Initialize our Trainer
    # training_args.gradient_checkpointing = False
    training_args.run_name = optimization_level

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=training_args.per_device_train_batch_size,
        drop_last=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        batch_size=training_args.per_device_eval_batch_size,
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
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
        lr=training_args.learning_rate,
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / training_args.gradient_accumulation_steps
    )
    # if training_args.max_train_steps is None:
    max_train_steps = int(training_args.num_train_epochs) * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=training_args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = (
        accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / training_args.gradient_accumulation_steps
    )

    # Afterwards we recalculate our number of training epochs
    training_args.num_train_epochs = math.ceil(
        max_train_steps / num_update_steps_per_epoch
    )

    # Figure out how many steps we should save the Accelerator states
    save_steps = training_args.save_steps

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.

    accelerator.init_trackers(
        "llm_pretraining_optimization",
        training_args,
        init_kwargs={"wandb": {"name": f"{optimization_level}_acc"}},
    )

    # Train!
    total_batch_size = (
        training_args.per_device_train_batch_size
        * accelerator.num_processes
        * training_args.gradient_accumulation_steps
    )
    print("total_batch_size", total_batch_size)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}"
    )
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    completed_steps = 0
    starting_epoch = 0
    log_steps = 0

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    # print(next(iter(train_dataloader)))
    # exit()
    global_step = 0
    total_loss = 0
    # print(model)
    model.train()
    model.zero_grad()
    for epoch in range(starting_epoch, training_args.num_train_epochs):
        active_dataloader = train_dataloader
        active_dataloader_len = len(active_dataloader)
        print("active_dataloader=", active_dataloader_len)
        active_dataloader.set_epoch(epoch)
        for local_step, batch in enumerate(active_dataloader):
            # with accelerator.no_sync(model):
            outputs = model(**batch)
            loss = outputs.loss

            outputs = None
            batch = None
            # We keep track of the loss at each epoch
            accelerator.backward(loss)
            total_loss += loss.detach().float()
            # так как gradient_accumulation_steps=1 в данном примере, то мы делаем
            # клиппинг каждый шаг
            _grad_norm = accelerator.clip_grad_norm_(
                model.parameters(),
                1.0,
            )
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if (global_step + 1) % training_args.logging_steps == 0:
                accelerator.log(
                    {
                        "train/loss": total_loss / training_args.logging_steps,
                        "train/learning_rate": lr_scheduler.get_last_lr()[0],
                        "train/grad_norm": _grad_norm,
                    },
                    step=log_steps,
                )
                log_steps += 1
                total_loss -= total_loss

            global_step += 1
            # if global_step > 30:
            #     break

    model = model.eval()
    # torch.compiler.cudagraph_mark_step_begin()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss.detach().float()
        outputs = None
        batch = None
        losses.append(
            accelerator.gather_for_metrics(
                loss.repeat(training_args.per_device_eval_batch_size)
            )
        )

    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")

    logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

    accelerator.log(
        {
            "eval/perplexity": perplexity,
            "eval/loss": eval_loss,
        },
        step=completed_steps,
    )

    output_dir = f"step_{completed_steps}"
    if training_args.output_dir is not None:
        output_dir = os.path.join(training_args.output_dir, output_dir)
    accelerator.save_state(output_dir)

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
