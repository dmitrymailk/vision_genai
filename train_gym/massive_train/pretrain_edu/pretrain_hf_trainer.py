import logging
import math
import os

os.environ["WANDB_PROJECT"] = "llm_pretraining"
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import evaluate
import torch
from datasets import load_dataset

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
)
from transformers.testing_utils import CaptureLogger
from transformers.utils.versions import require_version

# from liger_kernel.transformers.functional import liger_cross_entropy
from typing import Any, Sequence, cast

# from cut_cross_entropy.transformers import cce_patch
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

from liger_kernel.transformers import AutoLigerKernelForCausalLM
from types import MethodType

# from torchao.float8 import convert_to_float8_training, Float8LinearConfig
from accelerate.utils import FP8RecipeKwargs
from torch.utils.data import (
    DataLoader,
    Dataset,
    IterableDataset,
    RandomSampler,
    SequentialSampler,
)
import random
from train_gym.massive_train.evaluation.custom_lm_eval_v2 import SimpleAccelerateHFLM
from lm_eval import evaluator
import torch.distributed as dist
from streaming import StreamingDataset
from transformers.trainer_utils import speed_metrics, get_last_checkpoint
import time


class PretrainTrainer(Trainer):
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

    def get_train_dataloader(self) -> DataLoader:
        # for mosaic streaming
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        return self.train_dataset

    def _evaluate(self, *args, **kwargs):
        self.model.eval()
        eval_model = SimpleAccelerateHFLM(
            pretrained=self.model,
            accelerator=self.accelerator,
            tokenizer=self.processing_class,
            config=self.model.config,
            batch_size=self.args.per_device_eval_batch_size,
        )
        target_metrics = [
            "arc_easy",
            "hellaswag",
            "winogrande",
            "sciq",
            "copa",
            "openbookqa",
            "mmlu_stem",
            "mmlu_other",
            "mmlu_social_sciences",
            "mmlu_humanities",
        ]
        metrics_result = evaluator.simple_evaluate(
            model=eval_model,
            tasks=target_metrics,
            verbosity="WARNING",
            batch_size=self.args.per_device_eval_batch_size,
        )
        self.accelerator.wait_for_everyone()
        report_dict = {}
        if self.accelerator.is_main_process:
            metrics_result = metrics_result["results"]
            # print(metrics_result)
            for metric_name in target_metrics:
                for key, value in metrics_result[metric_name].items():
                    report_dict[f"eval_{metric_name}_{key}"] = value
            self.log(report_dict)
        # данный объект используется например для отбора лучшего чекпоинта, поэтому его нужно передать дальше
        # по всем потокам
        metrics_to_broadcast = [report_dict]
        dist.broadcast_object_list(metrics_to_broadcast, src=0)
        synced_lm_eval_metrics = metrics_to_broadcast[0]

        return synced_lm_eval_metrics


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    attn_implementation: Optional[str] = field(
        default="eager",
        metadata={
            "choices": ["eager", "sdpa", "flash_attention_2"],
        },
    )
    optimization_level: Optional[str] = field(
        default="opt_1",
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataloader_type: Optional[str] = field(
        default="hf_edu",
        metadata={
            "help": "The type dataloader_type (default hf method or mosaic streaming)"
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True,
        metadata={"help": "Whether to keep line breaks when using TXT files or not."},
    )


logger = logging.getLogger(__name__)


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
    dataloader_type = data_args.dataloader_type
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
    match optimization_level:
        case "opt_1":
            print("opt_1")
            # https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.from_pretrained.attn_implementation
            # model = AutoModelForCausalLM.from_pretrained(
            #     model_name_or_path,
            #     torch_dtype=torch_dtype,
            #     attn_implementation=model_args.attn_implementation,
            # )
            model = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=torch_dtype,
                attn_implementation=model_args.attn_implementation,
            )

    match dataloader_type:
        case "hf_edu":
            tok_logger = transformers.utils.logging.get_logger(
                "transformers.tokenization_utils_base"
            )
            print("load hf_edu dataset")
            raw_datasets = load_dataset(
                "HuggingFaceFW/fineweb-edu",
                name="sample-10BT",
                split="train",
                cache_dir="/code/fineweb_edu_10b",
                num_proc=16,
            )
            raw_datasets = raw_datasets.select(list(range(len(raw_datasets) // 6)))

            # column_names = list(raw_datasets["train"].features)
            column_names = list(raw_datasets.features)
            text_column_name = "text" if "text" in column_names else column_names[0]
            # block_size = 2048
            block_size = data_args.block_size

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

            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=min(os.cpu_count(), 32),
                remove_columns=column_names,
                # load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

            def group_texts(examples):
                # Concatenate all texts.
                concatenated_examples = {
                    k: list(chain(*examples[k])) for k in examples.keys()
                }
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
                # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
                total_length = (total_length // block_size) * block_size
                # Split by chunks of max_len.
                result = {
                    k: [
                        t[i : i + block_size]
                        for i in range(0, total_length, block_size)
                    ]
                    for k, t in concatenated_examples.items()
                }
                result["labels"] = result["input_ids"].copy()
                return result

            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=min(os.cpu_count(), 32),
                desc=f"Grouping texts in chunks of {block_size}",
            )

            lm_datasets = lm_datasets.remove_columns(
                column_names=[
                    item
                    for item in lm_datasets.features.keys()
                    if not item
                    in [
                        "input_ids",
                        "labels",
                        "attention_mask",
                    ]
                ]
            )
            train_dataset = lm_datasets
        case "mosaic_edu":
            from streaming.base.util import clean_stale_shared_memory

            clean_stale_shared_memory()
            # local_dir = "fineweb_edu_10b_numpy_mds_chunked"
            # local_dir = "/code/fineweb_edu_10b_numpy_mds_chunked"
            # run rm -rf /dev/shm/* if stuck
            # local_dir = "/code/fineweb_edu_10b_numpy_mds_chunked_1024"
            local_dir = "/code/fineweb_edu_10b_numpy_mds_chunked_2048"
            train_dataset = StreamingDataset(
                local=local_dir,
                remote=local_dir,
                batch_size=training_args.per_device_train_batch_size,
                # batch_size=64,
                split=None,
                shuffle=True,
            )
            train_dataset = DataLoader(
                train_dataset,
                batch_size=training_args.per_device_train_batch_size,
                pin_memory=True,
                num_workers=4,
                collate_fn=default_data_collator,
                drop_last=True,
                # shuffle=True,
                # persistent_workers=True,
            )

    # print(training_args)
    # Initialize our Trainer
    training_args.gradient_checkpointing = False
    training_args.run_name = optimization_level
    # 176_291_840
    # 010_000_000
    save_tokens = 20_000_000
    world_size = torch.cuda.device_count()
    save_steps = save_tokens // (
        data_args.block_size * training_args.per_device_train_batch_size * world_size
    )
    print("save_steps/eval_steps", save_steps)
    training_args.save_steps = save_steps
    training_args.eval_steps = save_steps

    # print(training_args.accelerator_config)
    trainer = PretrainTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    resume_from_checkpoint = get_last_checkpoint(training_args.output_dir)
    resume_from_checkpoint = not resume_from_checkpoint is None

    # Training
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload


if __name__ == "__main__":
    main()
