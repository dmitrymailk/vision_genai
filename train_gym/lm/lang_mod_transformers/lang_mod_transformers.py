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


logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

from lang_mod_transformers.utils import (
    ModelArguments,
    DataTrainingArguments,
    cuda_streams_forward,
)
from liger_kernel.transformers import AutoLigerKernelForCausalLM
from types import MethodType
from torchao.float8 import convert_to_float8_training, Float8LinearConfig


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
    match optimization_level:
        case "opt_1":
            print("opt_1")
            # https://huggingface.co/docs/transformers/en/main_classes/model#transformers.PreTrainedModel.from_pretrained.attn_implementation
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                attn_implementation=model_args.attn_implementation,
            )
        case "opt_2":
            print("opt_2")
            model = AutoLigerKernelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                use_cache=False,
                torch_dtype=torch_dtype,
            )
        case "opt_3":
            print("opt_3")
            model = AutoLigerKernelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                use_cache=False,
                torch_dtype=torch_dtype,
                cross_entropy=True,
                fused_linear_cross_entropy=False,
            )
        case "opt_4":
            print("opt_4")
            model = AutoLigerKernelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                use_cache=False,
                torch_dtype=torch_dtype,
                cross_entropy=False,
                fused_linear_cross_entropy=True,
            )
        case "opt_5":
            print("opt_5")

            nn.functional.cross_entropy = liger_cross_entropy
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                attn_implementation=model_args.attn_implementation,
            )
        case "opt_6":
            print("opt_6")
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                attn_implementation=model_args.attn_implementation,
            )
            model = cast(transformers.PreTrainedModel, model)
            cross_entropy_impl = "cce"
            # original apple implementation
            model = cce_patch(
                model,
                cross_entropy_impl,
                train_only=True,
            )
        case "opt_7":
            print("opt_7")
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                attn_implementation=model_args.attn_implementation,
            )
            model = cast(transformers.PreTrainedModel, model)
            cross_entropy_impl = "cce"
            # unsloth
            model = cce_patch(
                model,
                cross_entropy_impl,
            )
        case "opt_8":
            print("opt_8")
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                attn_implementation=model_args.attn_implementation,
            )
            # packed sequence with flash attention
            data_collator = DataCollatorWithFlattening()
        case "opt_9":
            print("opt_9")
            model = AutoLigerKernelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                use_cache=False,
                torch_dtype=torch_dtype,
            )
            # unsloth
            cross_entropy_impl = "cce"
            model = cce_patch(
                model,
                cross_entropy_impl,
            )
        case "opt_10":
            print("opt_10")
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                use_cache=False,
                torch_dtype=torch_dtype,
            )
            model = torch.compile(
                model,
                backend="inductor",
            )
        case "opt_11":
            print("opt_11")
            model = AutoLigerKernelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                use_cache=False,
                torch_dtype=torch_dtype,
            )
            # unsloth
            cross_entropy_impl = "cce"
            model = cce_patch(
                model,
                cross_entropy_impl,
            )
            model = torch.compile(
                model,
                backend="inductor",
            )
        case "opt_12":
            print("opt_12")
            model = AutoLigerKernelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                use_cache=False,
                torch_dtype=torch_dtype,
                cross_entropy=True,
                fused_linear_cross_entropy=False,
            )
            model = torch.compile(
                model,
                backend="inductor",
            )
        case "opt_13":
            print("opt_13")
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                use_cache=False,
                torch_dtype=torch_dtype,
            )
            for m in reversed(list(model.modules())):
                if isinstance(m, LlamaAttention):
                    m.compile(backend="inductor")
        case "opt_14":
            print("opt_14")
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                use_cache=False,
                torch_dtype=torch_dtype,
            )
            for m in reversed(list(model.modules())):
                if isinstance(m, LlamaDecoderLayer):
                    m.compile(backend="inductor")
        case "opt_15":
            print("opt_15")
            model = AutoLigerKernelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                use_cache=False,
                torch_dtype=torch_dtype,
            )
            # unsloth
            cross_entropy_impl = "cce"
            model = cce_patch(
                model,
                cross_entropy_impl,
            )

            for m in reversed(list(model.modules())):
                if isinstance(m, LlamaDecoderLayer):
                    m.compile(
                        backend="inductor",
                    )
        case "opt_16":
            print("opt_16")
            model = AutoLigerKernelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                use_cache=False,
                torch_dtype=torch_dtype,
            )
            # unsloth
            cross_entropy_impl = "cce"
            model = cce_patch(
                model,
                cross_entropy_impl,
            )

            for m in reversed(list(model.modules())):
                if isinstance(m, LlamaDecoderLayer):
                    m.compile(
                        backend="inductor",
                        mode="max-autotune",
                    )
        case "opt_17":
            print("opt_17")
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                use_cache=False,
                torch_dtype=torch_dtype,
            )

            for m in reversed(list(model.modules())):
                if isinstance(m, LlamaDecoderLayer):
                    m.compile(
                        backend="inductor",
                        mode="max-autotune",
                    )
        case "opt_18":
            print("opt_18")
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                use_cache=False,
                torch_dtype=torch_dtype,
            )
            # unsloth
            cross_entropy_impl = "cce"
            model = cce_patch(
                model,
                cross_entropy_impl,
            )

            for m in reversed(list(model.modules())):
                if isinstance(m, LlamaDecoderLayer):
                    m.compile(
                        backend="inductor",
                        mode="max-autotune",
                    )
        case "opt_19":
            print("opt_19")
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                use_cache=False,
                torch_dtype=torch_dtype,
            )
            model = torch.compile(
                model,
                backend="inductor",
                mode="max-autotune",
            )
        case "opt_20":
            print("opt_20")
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                use_cache=False,
                torch_dtype=torch_dtype,
            )
            cuda_streams = [torch.cuda.Stream() for _ in model.model.layers]
            model.model.cuda_streams = cuda_streams
            LlamaModel.forward = cuda_streams_forward
            # unsloth
            cross_entropy_impl = "cce"
            model = cce_patch(
                model,
                cross_entropy_impl,
            )

            for m in reversed(list(model.modules())):
                if isinstance(m, LlamaDecoderLayer):
                    m.compile(
                        backend="inductor",
                        mode="max-autotune",
                        fullgraph=True,
                    )
        case "opt_21":
            print("opt_21")
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                use_cache=False,
                torch_dtype=torch_dtype,
            )
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
                if isinstance(m, LlamaDecoderLayer):
                    m.compile(
                        backend="inductor",
                        # mode="max-autotune",
                    )
        case "opt_22":
            print("opt_22")
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                use_cache=False,
                torch_dtype=torch_dtype,
            )
            # unsloth
            cross_entropy_impl = "cce"
            model = cce_patch(
                model,
                cross_entropy_impl,
            )
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
                if isinstance(m, LlamaDecoderLayer):
                    m.compile(
                        backend="inductor",
                        # mode="max-autotune",
                    )
        case "opt_23":
            print("opt_23")
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                use_cache=False,
                torch_dtype=torch_dtype,
            )
            # unsloth
            cross_entropy_impl = "cce"
            model = cce_patch(
                model,
                cross_entropy_impl,
            )
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
            model = compile_regions(model)

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

    # block_size = 1024
    block_size = data_args.block_size

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
    training_args.gradient_checkpointing = False
    training_args.run_name = optimization_level
    # training_args.accelerator_config =
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        # data_collator=default_data_collator,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # Training
    train_result = trainer.train()
    # trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    # trainer.save_metrics("train", metrics)

    # Evaluation
    logger.info("*** Evaluate ***")
    torch.compiler.cudagraph_mark_step_begin()
    model.forward = MethodType(original_forward, model)
    metrics = trainer.evaluate()

    metrics["eval_samples"] = len(eval_dataset)
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity

    trainer.log_metrics("eval", metrics)
    # trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
