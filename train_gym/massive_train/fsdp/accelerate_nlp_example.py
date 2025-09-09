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

# from lang_mod_transformers.utils import (
#     ModelArguments,
#     DataTrainingArguments,
#     cuda_streams_forward,
# )
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
import time


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

# from lang_mod_transformers.llama3_2_hf_v2 import (
#     LlamaForCausalLM as LlamaForCausalLMHF_V2,
# )
# from lang_mod_transformers import llama3_2_hf_v2
from cut_cross_entropy.transformers.llama import (
    cce_forward,
    linear_cross_entropy,
    _PATCH_OPTS,
)

# from cut_cross_entropy.transformers.utils import PatchOptions
# from lang_mod_transformers.llama3_2_torchtune_v3 import (
#     llama3_2_1b,
#     hf_to_tune,
#     TransformerDecoder,
#     TransformerSelfAttentionLayer,
# )

# from torchtune.modules.optim import OptimizerInBackward
import bitsandbytes as bnb

# from
# from transformer_engine.common.recipe import DelayedScaling

# torch.backends.cudnn.allow_tf32 = True
# torch.backends.cuda.matmul.allow_tf32 = True

from transformers.utils.versions import require_version
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
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
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast
from streaming import MDSWriter, StreamingDataset
from accelerate.utils import DistributedDataParallelKwargs

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


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
            + ", ".join(MODEL_TYPES)
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
        default=None,
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

    def __post_init__(self):
        if self.streaming:
            require_version(
                "datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`"
            )

        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`validation_file` should be a csv, a json or a txt file."


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
    # pc = ParallelismConfig()
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
    accelerator_log_kwargs = {
        "log_with": "wandb",
        # "log_with": "trackio",
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

        case "opt_28":
            print("opt_28")

            model = LlamaForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                use_cache=False,
                torch_dtype=torch_dtype,
            )
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
            major, minor = torch.cuda.get_device_capability()
            # Target version
            target_major = 8
            target_minor = 9
            if (major > target_major) or (
                major == target_major and minor >= target_minor
            ):
                convert_to_float8_training(
                    model,
                    config=config,
                    module_filter_fn=func,
                )
            # model = torch.compile(model)

            for m in reversed(list(model.modules())):
                if isinstance(m, LlamaDecoderLayer):
                    m.compile(
                        backend="inductor",
                        # mode="max-autotune",
                    )

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

    print("model_args.attn_implementation", model_args.attn_implementation)

    match dataloader_type:
        case "mosaic_edu":
            # from streaming.base.util import clean_stale_shared_memory

            # clean_stale_shared_memory()
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
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=training_args.per_device_train_batch_size,
                pin_memory=True,
                num_workers=training_args.per_device_train_batch_size,
                collate_fn=default_data_collator,
                drop_last=True,
                # shuffle=True,
                persistent_workers=True,
            )

    # print(train_dataset[0])

    print(training_args)
    # Initialize our Trainer
    # training_args.gradient_checkpointing = False
    training_args.run_name = optimization_level

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
        fused=True,
    )
    # optimizer = bnb.optim.Adam8bit(
    #     optimizer_grouped_parameters,
    #     lr=training_args.learning_rate,
    # )

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
    # model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = (
    #     accelerator.prepare(
    #         model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    #     )
    # )
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
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
    total_tokens = 0
    # print(model)
    model.train()
    model.zero_grad()
    last_log_time = time.monotonic()
    last_log_total_tokens = 0
    for epoch in range(starting_epoch, training_args.num_train_epochs):
        active_dataloader = train_dataloader
        active_dataloader_len = len(active_dataloader)
        print("active_dataloader=", active_dataloader_len)
        active_dataloader.set_epoch(epoch)
        for local_step, batch in enumerate(active_dataloader):
            # with accelerator.no_sync(model):
            # batch["labels"] = batch["input_ids"].clone()
            # print("batch", batch)
            batch_tokens = batch["input_ids"].shape
            # print(batch_tokens)
            batch_toks = 1
            for item in batch_tokens:
                batch_toks *= item
            print(batch_toks)
            total_tokens += batch_toks
            outputs = model(**batch)
            loss = outputs.loss
            # print("loss", loss)

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
            accelerator.gradient_state._set_sync_gradients(True)
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if (global_step + 1) % training_args.logging_steps == 0:
                # Вместо этого считаем среднюю скорость с последнего лога
                current_time = time.monotonic()
                elapsed_time = current_time - last_log_time
                tokens_processed = total_tokens - last_log_total_tokens
                print(tokens_processed, tokens_processed / training_args.logging_steps)
                # Вот это и есть ваша реальная средняя пропускная способность
                effective_tokens_per_second = tokens_processed / elapsed_time
                print("effective_tokens_per_second", effective_tokens_per_second)
                accelerator.log(
                    {
                        "train/loss": total_loss / training_args.logging_steps,
                        "train/learning_rate": lr_scheduler.get_last_lr()[0],
                        "train/grad_norm": _grad_norm,
                        "throughput/device/tokens_per_second": effective_tokens_per_second,
                        "throughput/total_tokens": total_tokens,
                    },
                    step=log_steps,
                )
                log_steps += 1
                total_loss -= total_loss
                last_log_time = current_time
                last_log_total_tokens = total_tokens

            global_step += 1
            # if global_step > 30:
            #     break

    output_dir = f"step_{completed_steps}"
    if training_args.output_dir is not None:
        output_dir = os.path.join(training_args.output_dir, output_dir)
    accelerator.save_state(output_dir)

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
