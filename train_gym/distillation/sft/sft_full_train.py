import os

os.environ["WANDB_PROJECT"] = "rmt_sft"


import torch

from datasets import load_dataset
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

from typing import List, Dict, Any
from torch.utils.data import DataLoader
from accelerate.utils import send_to_device
from transformers import AutoTokenizer

import argparse
import os
from typing import Optional

from accelerate import logging
from datasets import load_dataset
from transformers import AutoConfig, AutoModelForCausalLM


from trl import (
    DatasetMixtureConfig,
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_dataset,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from liger_kernel.transformers import apply_liger_kernel_to_qwen2

from datasets import Dataset, IterableDataset
from transformers import (
    AutoConfig,
    AutoProcessor,
    BaseImageProcessor,
    DataCollator,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainingArguments,
    is_wandb_available,
)
from typing import Any, Callable, Optional, TypeVar, Union
from accelerate import PartialState, logging
from trl.trainer.sft_trainer import remove_none_values
from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    is_conversational_from_value,
    maybe_convert_to_chatml,
    pack_dataset,
    prepare_multimodal_messages,
    truncate_dataset,
)
from transformers.trainer_utils import speed_metrics, get_last_checkpoint
import gc
import pandas as pd
import copy
from random import shuffle
import json
from train_gym.distillation.sft.utils import (
    train_on_responses_only,
    print_trainable_parameters,
)
from lm_eval import evaluator
import time
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np
import wandb
from transformers.integrations import WandbCallback
import matplotlib
from train_gym.massive_train.evaluation.custom_lm_eval_v2 import SimpleAccelerateHFLM
import torch.distributed as dist
from torchao.float8 import convert_to_float8_training, Float8LinearConfig
from functools import partial
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
from dataclasses import dataclass
from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3
from train_gym.rmt.rmt_wrappers import (
    ChatMemoryCellTrainV2,
    MemoryCell,
    RecurrentWrapper,
    MemoryCellTrain,
    RecurrentWrapperTrain,
    MemoryCellTrainLiger,
    RecurrentWrapperTrainV2,
    lce_forward,
    ChatMemoryCellTrain,
)
from types import MethodType
from peft import LoraConfig, get_peft_model
from transformers.utils import is_peft_available
import matplotlib

matplotlib.use("Agg")

if is_peft_available():
    from peft import PeftModel

logger = logging.get_logger(__name__)

TRAINING_ARGS_NAME = "training_args.bin"
MAX_SEQ_LENGTH = 2240


@dataclass
class RMTArguments:
    opt_version: str = "opt_1"
    apply_rmt: bool = False
    memory_size: int = 32
    segment_size: int = 1024
    max_n_segments: int = 2
    vary_n_segments: bool = False
    k2: int = -1


def filter_linear_layers(module, fqn, first_layer_name=None, last_layer_name=None):
    if isinstance(module, torch.nn.Linear):
        if module.in_features % 16 != 0 or module.out_features % 16 != 0:
            return False
    # For stability reasons, we skip the first and last linear layers
    # Otherwise can lead to the model not training or converging properly
    if fqn in (first_layer_name, last_layer_name):
        return False
    return True


def make_parser(subparsers: Optional[argparse._SubParsersAction] = None):
    dataclass_types = (
        ScriptArguments,
        SFTConfig,
        ModelConfig,
        DatasetMixtureConfig,
        RMTArguments,
    )
    if subparsers is not None:
        parser = subparsers.add_parser(
            "sft", help="Run the SFT training script", dataclass_types=dataclass_types
        )
    else:
        parser = TrlParser(dataclass_types)
    return parser


def get_conversation_length(item, tokenizer):
    if "messages" in item:
        messages = item["messages"]
    elif "conversations" in item:
        messages = []
        for mess in item["conversations"]:
            value = mess["value"]
            from_ = mess["from"]
            if from_ == "gpt":
                from_ = "assistant"
            messages.append(
                {
                    "role": from_,
                    "content": value,
                }
            )
    else:
        messages = []
        for mess in item["conversation"]:
            value = mess["content"]
            from_ = mess["role"]

            messages.append(
                {
                    "role": from_,
                    "content": value,
                }
            )

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    length = len(
        tokenizer.encode(
            text,
            add_special_tokens=False,
        )
    )
    return {"length": length}


class ChatGenerationDataset(torch.utils.data.Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        item = self.dataset[idx]
        if "messages" in item:
            messages = item["messages"]
        elif "conversations" in item:
            messages = []
            for mess in item["conversations"]:
                value = mess["value"]
                from_ = mess["from"]
                if from_ == "gpt":
                    from_ = "assistant"
                messages.append(
                    {
                        "role": from_,
                        "content": value,
                    }
                )
        else:
            messages = []
            for mess in item["conversation"]:
                value = mess["content"]
                from_ = mess["role"]

                messages.append(
                    {
                        "role": from_,
                        "content": value,
                    }
                )

        return {
            "messages": messages,
        }


class DeviceDataLoader(DataLoader):
    def __iter__(self):
        cpu_iterator = super().__iter__()
        current_device = torch.cuda.current_device()
        current_device = torch.device(f"cuda:{current_device}")
        for batch in cpu_iterator:
            yield send_to_device(batch, current_device)


class DataCollatorForGeneration:
    def __init__(
        self,
        tokenizer,
        args: SFTConfig = None,
    ):
        self.tokenizer = tokenizer

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.args = args

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # print(features)
        prompts = [f["messages"] for f in features]

        processed = self.tokenizer.apply_chat_template(
            prompts,
            return_dict=True,
            # return_assistant_tokens_mask=self.args.assistant_only_loss,
            # tools=example.get("tools"),
            # **example.get("chat_template_kwargs", {}),
            tokenize=True,
            # add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )
        # print(processed)
        if "assistant_masks" in processed and 1 not in processed["assistant_masks"]:
            raise RuntimeError(
                "You're using `assistant_only_loss=True`, but at least one example has no "
                "assistant tokens. This usually means the tokenizer's chat template doesn't "
                "generate assistant masks — it may be missing the `{% generation %}` keyword. Please "
                "check the template and ensure it's correctly configured to support assistant "
                "masking."
            )
        # output = {
        #     k: processed[k]
        #     for k in ("input_ids", "assistant_masks")
        #     if k in processed
        # }
        # мы всегда обучаемся только на продолжении
        processed["labels"] = processed["input_ids"].clone()

        post_prosess_func = train_on_responses_only(
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
            tokenizer=self.tokenizer,
        )
        processed["labels"] = post_prosess_func(processed)
        processed["labels"] = processed["labels"]["labels"]

        processed["labels"][processed["labels"] == self.tokenizer.pad_token_id] = -100

        if self.args.assistant_only_loss:
            processed["labels"][processed["assistant_masks"] == 0] = -100
        return processed


class EvalSFTTrainer(SFTTrainer):
    def _prepare_dataset(
        self,
        dataset: Union[Dataset, IterableDataset],
        *args,
        **kwargs,
    ) -> Union[Dataset, IterableDataset]:
        return dataset

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            train_metadata = getattr(self.state, "train_metadata", None)

            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            input_tokens_seen = self.state.num_input_tokens_seen
            if not train_metadata is None:
                start_time = train_metadata["prev_log_time"]
                input_tokens_seen = (
                    input_tokens_seen - train_metadata["prev_total_tokens"]
                )
                # мы полагаемся только на локальное время, потому что если мы передадим время
                # из главного цикла, но будем продолжать обучение с чекпоинта у нас будет выброс
                # по скорости
                logs.update(
                    speed_metrics("train", start_time, num_tokens=input_tokens_seen)
                )

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(
            self.args, self.state, self.control, logs
        )

        # чтобы не добавлять новых полей в state и не переписывать всё, просто сохраним в одно поле
        if self.args.include_num_input_tokens_seen:
            setattr(
                self.state,
                "train_metadata",
                {
                    "prev_total_tokens": self.state.num_input_tokens_seen,
                    "prev_log_time": time.time(),
                },
            )

    def create_babilong_plot(
        self,
        report_dict,
        model_name="",
        dataset_name="babilongv2",
    ):
        rows = ["qa1", "qa2", "qa3", "qa4", "qa5"]
        cols = ["0k", "1k", "2k", "4k"]

        babilong_matrix = np.zeros((len(rows), len(cols)))

        for key in report_dict:
            if dataset_name in key:
                for i, row in enumerate(rows):
                    for j, col in enumerate(cols):
                        if row in key and col in key:
                            babilong_matrix[i, j] = report_dict[key]

                cmap = LinearSegmentedColormap.from_list(
                    "ryg", ["red", "yellow", "green"], N=256
                )

        fig, ax = plt.subplots(1, 1, figsize=(5 * 1, 3.5))
        sns.heatmap(
            babilong_matrix * 100,
            cmap=cmap,
            vmin=0,
            vmax=100,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            xticklabels=cols,
            yticklabels=rows,
            ax=ax,
        )
        ax.set_title(f"{model_name}", pad=20, y=0.95)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        pil_image = Image.open(buf)
        return pil_image

    def _evaluate(self, *args, **kwargs):
        self.model.eval()
        train_metadata = getattr(self.state, "train_metadata", None)

        if (
            train_metadata is None
            and self.state.global_step == 0
            or not train_metadata is None
            and self.state.global_step > 0
        ):
            gc.collect()
            torch.cuda.empty_cache()
            self.accelerator.wait_for_everyone()

            eval_model = SimpleAccelerateHFLM(
                pretrained=self.model,
                accelerator=self.accelerator,
                tokenizer=self.processing_class,
                config=self.model.config,
                batch_size=self.args.per_device_eval_batch_size,
                max_length=4096,
            )
            eval_metrics = [
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
                "babilongv2_qa1_under_4k_instruct",
                # "babilongv2_qa2_under_4k_instruct",
                # "babilongv2_qa3_under_4k_instruct",
                # "babilongv2_qa4_under_4k_instruct",
                # "babilongv2_qa5_under_4k_instruct",
                # for debugging
                # "babilongv2_qa1_0k_instruct"
            ]
            target_metrics = [
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
                "babilongv2_qa1_0k_instruct",
                "babilongv2_qa1_1k_instruct",
                "babilongv2_qa1_2k_instruct",
                "babilongv2_qa1_4k_instruct",
                # "babilongv2_qa2_0k_instruct",
                # "babilongv2_qa2_1k_instruct",
                # "babilongv2_qa2_2k_instruct",
                # "babilongv2_qa2_4k_instruct",
                # "babilongv2_qa3_0k_instruct",
                # "babilongv2_qa3_1k_instruct",
                # "babilongv2_qa3_2k_instruct",
                # "babilongv2_qa3_4k_instruct",
                # "babilongv2_qa4_0k_instruct",
                # "babilongv2_qa4_1k_instruct",
                # "babilongv2_qa4_2k_instruct",
                # "babilongv2_qa4_4k_instruct",
                # "babilongv2_qa5_0k_instruct",
                # "babilongv2_qa5_1k_instruct",
                # "babilongv2_qa5_2k_instruct",
                # "babilongv2_qa5_4k_instruct",
                # for debugging
                # "babilongv2_qa1_0k_instruct",
            ]
            metrics_result = evaluator.simple_evaluate(
                model=eval_model,
                tasks=eval_metrics,
                verbosity="WARNING",
                batch_size=self.args.per_device_eval_batch_size,
                # limit=100,
            )
            gc.collect()
            torch.cuda.empty_cache()
            self.accelerator.wait_for_everyone()
            report_dict = {}
            if self.accelerator.is_main_process:
                metrics_result = metrics_result["results"]

                ban_keys = ["stderr", "alias", "under"]
                for metric_name in target_metrics:
                    for key, value in metrics_result[metric_name].items():
                        eval_key = f"eval_{metric_name}_{key}"
                        if not any(ban_key in eval_key for ban_key in ban_keys):
                            report_dict[eval_key] = value

                babilong_name = "babilongv2"
                babilong_plot = self.create_babilong_plot(
                    report_dict=report_dict,
                    model_name=babilong_name,
                )
                for callback in self.callback_handler.callbacks:
                    if isinstance(callback, WandbCallback):
                        # мы не можем логгировать изображения данным трейнером
                        # потому что картинка не json serializable
                        callback._wandb.log(
                            {
                                f"plots/{babilong_name}": wandb.Image(babilong_plot),
                            },
                        )

            metrics_to_broadcast = [report_dict]
            if torch.cuda.device_count() > 1:
                dist.broadcast_object_list(metrics_to_broadcast, src=0)
            synced_lm_eval_metrics = metrics_to_broadcast[0]
            self.log(synced_lm_eval_metrics)

            return synced_lm_eval_metrics

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        rmt_classes = RecurrentWrapperTrain
        if isinstance(self.model, rmt_classes):
            self.model.memory_cell.model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=self.args.save_safetensors,
            )
            torch.save(self.model.memory_cell.memory, f"{output_dir}/memory.pt")

        else:
            self.model.save_pretrained(
                output_dir,
                state_dict=state_dict,
                safe_serialization=self.args.save_safetensors,
            )

        if self.processing_class is not None:
            self.processing_class.save_pretrained(output_dir)
        elif (
            self.data_collator is not None
            and hasattr(self.data_collator, "tokenizer")
            and self.data_collator.tokenizer is not None
        ):
            logger.info(
                "Saving Trainer.data_collator.tokenizer by default as Trainer.processing_class is `None`"
            )
            self.data_collator.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))


DATASET_PARAMS = {
    "HuggingFaceH4/ultrachat_200k": {
        "cache_dir": "/code/ultrachat_200k",
        "field": "train_sft",
        "length": 2240,
    },
    "zai-org/LongAlign-10k": {
        "cache_dir": "/code/datasets/LongAlign-10k",
        "field": "train",
        "length": 43141,
    },
    "allenai/WildChat-4.8M": {
        "cache_dir": "/code/datasets/WildChat-4.8M",
        "field": "train",
        "length": 4964,
    },
}

if __name__ == "__main__":

    parser = make_parser()
    script_args, training_args, model_args, dataset_args, rmt_args, _ = (
        parser.parse_args_and_config(return_remaining_strings=True)
    )
    if training_args.use_liger_kernel:
        print("apply use_liger_kernel")
        apply_liger_kernel_to_qwen3()

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.padding_side = "right"

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=model_args.dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )
    # liger kernel slower
    # model.forward = MethodType(lce_forward, model)
    model.config.use_cache = False
    if rmt_args.apply_rmt:
        print("apply_rmt")
        # посегментное вычисление лосса

        # isk but with liger it's slower
        # cell = MemoryCellTrainLiger(
        # cell = MemoryCellTrain(
        match rmt_args.opt_version:
            case "opt_1":
                cell = ChatMemoryCellTrain(
                    model,
                    num_mem_tokens=rmt_args.memory_size,
                )
                model = RecurrentWrapperTrain(
                    cell,
                    segment_size=rmt_args.segment_size,
                    max_n_segments=rmt_args.max_n_segments,
                    vary_n_segments=rmt_args.vary_n_segments,
                    k2=rmt_args.k2,
                )
            case "opt_2":
                cell = ChatMemoryCellTrainV2(
                    model,
                    num_mem_tokens=rmt_args.memory_size,
                )
                model = RecurrentWrapperTrainV2(
                    cell,
                    segment_size=rmt_args.segment_size,
                    max_n_segments=rmt_args.max_n_segments,
                    vary_n_segments=rmt_args.vary_n_segments,
                    k2=rmt_args.k2,
                )

    print_trainable_parameters(model)

    dataset_name = script_args.dataset_name
    dataset = load_dataset(
        dataset_name,
        cache_dir=DATASET_PARAMS[dataset_name]["cache_dir"],
    )
    dataset = dataset[DATASET_PARAMS[dataset_name]["field"]]
    if dataset_name == "allenai/WildChat-4.8M":
        dataset = dataset.filter(
            lambda example: not "gpt-3.5" in example["model"],
            num_proc=training_args.dataset_num_proc,
        )
        # ).select(range(50000))

    dataset = dataset.map(
        lambda item: get_conversation_length(
            item,
            tokenizer=tokenizer,
        ),
        num_proc=training_args.dataset_num_proc,
    )
    dataset = dataset.filter(
        lambda example: example["length"]
        < DATASET_PARAMS[script_args.dataset_name]["length"],
        num_proc=training_args.dataset_num_proc,
    )
    MAX_SEQ_LENGTH = DATASET_PARAMS[script_args.dataset_name]["length"]

    dataset = dataset.train_test_split(
        dataset_args.test_split_size,
        seed=42,
    )
    eval_dataset = ChatGenerationDataset(
        dataset=dataset["test"],
    )

    train_dataset = ChatGenerationDataset(
        dataset=dataset["train"],
    )
    train_collator = DataCollatorForGeneration(
        tokenizer=tokenizer,
        args=training_args,
    )

    training_args.max_length = MAX_SEQ_LENGTH
    training_args.run_name = training_args.output_dir
    one_epoch_steps = (
        len(dataset["train"])
        // training_args.per_device_train_batch_size
        // training_args.gradient_accumulation_steps
        * 2
    )
    print("one_epoch_steps=", one_epoch_steps)
    training_args.max_steps = one_epoch_steps
    trainer = EvalSFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        data_collator=train_collator,
        args=training_args,
    )

    setattr(trainer, "sft_eval_dataset", eval_dataset)

    # resume_from_checkpoint = os.path.exists(training_args.output_dir)
    # if "checkpoint" in model_args.model_name_or_path:
    #     resume_from_checkpoint = True
    trainer_stats = trainer.train()
