import logging
import os

os.environ["WANDB_PROJECT"] = "llm_pretraining"
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional
import datasets
import torch
from datasets import load_dataset
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from functools import partial
from types import MethodType
from train_gym.massive_train.evaluation.custom_lm_eval_v2 import SimpleAccelerateHFLM
from lm_eval import evaluator
import torch.distributed as dist
from streaming import StreamingDataset
from transformers.trainer_utils import speed_metrics, get_last_checkpoint
import time
import gc
from cut_cross_entropy.transformers.llama import cce_forward
from torchao.float8 import convert_to_float8_training, Float8LinearConfig
import argparse
import torch
from liger_kernel.transformers import apply_liger_kernel_to_llama
import liger_kernel
from train_gym.massive_train.pretrain_edu.pretrain_hf_trainer import (
    PretrainTrainer,
    DataTrainingArguments,
    filter_linear_layers,
)
from train_gym.rmt.rmt_wrappers import (
    MemoryCell,
    RecurrentWrapper,
    MemoryCellTrain,
    RecurrentWrapperTrain,
    MemoryCellTrainLiger,
    lce_forward,
)


@dataclass
class ModelArguments:

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    optimization_level: Optional[str] = field(
        default="opt_1",
        metadata={
            "help": ("The model optimization variant."),
        },
    )
    memory_size: Optional[int] = field(
        default=32,
        metadata={
            "help": ("rmt memory size"),
        },
    )
    max_n_segments: Optional[int] = field(
        default=2,
        metadata={
            "help": ("rmt max segments"),
        },
    )
    k2: Optional[int] = field(
        default=-1,
        metadata={
            "help": ("k2"),
        },
    )
    segment_size: Optional[int] = field(
        default=1024,
        metadata={
            "help": ("segment_size"),
        },
    )
    vary_n_segments: Optional[bool] = field(
        default=False,
        metadata={"help": "vary_n_segments"},
    )


logger = logging.getLogger(__name__)


def main():

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    cli_parser = argparse.ArgumentParser()
    cli_parser.add_argument(
        "--yaml_file",
        type=str,
        default="/code/train_gym/massive_train/pretrain_edu/configs/llama3.2-1B.yaml",
        required=True,
        help="yaml_file input file to process.",
    )
    yaml_file = cli_parser.parse_args().yaml_file

    model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=yaml_file)

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
    config = AutoConfig.from_pretrained(
        model_name_or_path,
    )

    match optimization_level:
        case "opt_1_rmt":
            print("opt_1_rmt")
            model = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
            )
            # посегментное вычисление лосса
            cell = MemoryCellTrain(
                model,
                num_mem_tokens=model_args.memory_size,
            )
            model = RecurrentWrapperTrain(
                cell,
                segment_size=model_args.segment_size,
                max_n_segments=model_args.max_n_segments,
                vary_n_segments=model_args.vary_n_segments,
                k2=model_args.k2,
            )
        case "opt_2_rmt":
            print("opt_2_rmt")
            # baseline
            model = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
            )
            cell = MemoryCell(
                model,
                num_mem_tokens=model_args.memory_size,
            )
            model = RecurrentWrapper(
                cell,
                segment_size=model_args.segment_size,
                max_n_segments=model_args.max_n_segments,
                vary_n_segments=model_args.vary_n_segments,
                k2=model_args.k2,
            )
        case "opt_3_rmt":
            print("opt_3_rmt")
            # apply liger kernel
            liger_kernel.transformers.model.llama.lce_forward = lce_forward
            apply_liger_kernel_to_llama()
            model = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
            )
            cell = MemoryCellTrainLiger(
                model,
                num_mem_tokens=model_args.memory_size,
            )
            model = RecurrentWrapperTrain(
                cell,
                segment_size=model_args.segment_size,
                max_n_segments=model_args.max_n_segments,
                vary_n_segments=model_args.vary_n_segments,
                k2=model_args.k2,
            )

    match dataloader_type:
        case "mosaic_edu":
            # run rm -rf /dev/shm/* if stuck
            # local_dir = "/code/fineweb_edu_10b_numpy_mds_chunked_1024"
            local_dir = "/code/fineweb_edu_10b_numpy_mds_chunked_2048"
            world_size = torch.cuda.device_count()
            train_dataset = StreamingDataset(
                local=local_dir,
                remote=local_dir,
                batch_size=training_args.per_device_train_batch_size,
                # batch_size=None,
                # batch_size=64,
                split=None,
                shuffle=True,
                num_canonical_nodes=world_size,
            )
            # если не выставить это, процесс зависнет и обучения не будет
            training_args.accelerator_config.dispatch_batches = False

    training_args.gradient_checkpointing = False
    training_args.run_name = (
        f"{optimization_level}_batch_{training_args.per_device_train_batch_size}"
    )

    # 176_291_840
    # 010_000_000
    # 405_635_072
    save_tokens = 200_000_000
    world_size = torch.cuda.device_count()
    tokens_per_step = (
        data_args.block_size * training_args.per_device_train_batch_size * world_size
    )
    save_steps = save_tokens // tokens_per_step
    max_train_tokens = data_args.max_train_tokens
    max_steps = max_train_tokens // tokens_per_step + 1
    print("save_steps/eval_steps", save_steps)
    print("max_steps", max_steps)
    training_args.save_steps = save_steps
    training_args.eval_steps = save_steps
    training_args.max_steps = max_steps
    # lr from OLMO2 paper, bad convergence on small scale for this model, idk.
    # training_args.learning_rate = 6e-4
    # TODO: mu initialization from yulan mini. recheck eval code. add train from file config
    if training_args.lr_scheduler_type == "warmup_stable_decay":
        # Based on MiniCPM, decay 10% https://arxiv.org/pdf/2404.06395
        # https://huggingface.co/docs/transformers/v4.56.2/en/main_classes/optimizer_schedules#transformers.get_wsd_schedule
        # yulan mini тоже использовали данный scheduler https://arxiv.org/pdf/2412.17743
        max_steps = training_args.max_steps
        num_decay_steps = int(0.1 * max_steps)
        training_args.lr_scheduler_kwargs["num_decay_steps"] = num_decay_steps

    trainer = PretrainTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    resume_from_checkpoint = os.path.exists(training_args.output_dir)
    if resume_from_checkpoint:
        resume_from_checkpoint = get_last_checkpoint(training_args.output_dir)
        resume_from_checkpoint = not resume_from_checkpoint is None

    # Training
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    metrics = trainer._evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
