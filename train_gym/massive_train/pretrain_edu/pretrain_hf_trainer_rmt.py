import logging
import os

os.environ["WANDB_PROJECT"] = "llm_pretraining"
import sys
from dataclasses import dataclass, field
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
import transformers
from types import MethodType
from train_gym.massive_train.evaluation.custom_lm_eval_v2 import SimpleAccelerateHFLM
from lm_eval import evaluator
import torch.distributed as dist
from streaming import StreamingDataset
from transformers.trainer_utils import speed_metrics, get_last_checkpoint
import time
import gc
import argparse
import torch
from liger_kernel.transformers import apply_liger_kernel_to_llama

from train_gym.massive_train.pretrain_edu.pretrain_hf_trainer import (
    PretrainTrainer,
    DataTrainingArguments,
)
from train_gym.rmt.rmt_wrappers import (
    MemoryCell,
    RecurrentWrapper,
    MemoryCellTrain,
    RecurrentWrapperTrain,
    MemoryCellTrainLiger,
    lce_forward,
)
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np
import wandb
from transformers.integrations import WandbCallback
import matplotlib
from types import MethodType

matplotlib.use("Agg")


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


class PretrainRMTTrainer(PretrainTrainer):

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
            eval_model = SimpleAccelerateHFLM(
                pretrained=self.model,
                accelerator=self.accelerator,
                tokenizer=self.processing_class,
                config=self.model.config,
                batch_size=self.args.per_device_eval_batch_size,
            )
            eval_metrics = [
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
                "babilongv2_qa1_under_4k_base",
                "babilongv2_qa2_under_4k_base",
                "babilongv2_qa3_under_4k_base",
                "babilongv2_qa4_under_4k_base",
                "babilongv2_qa5_under_4k_base",
                # for debugging
                # "babilongv2_qa1_0k_base"
            ]
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
                "babilongv2_qa1_0k_base",
                "babilongv2_qa1_1k_base",
                "babilongv2_qa1_2k_base",
                "babilongv2_qa1_4k_base",
                "babilongv2_qa2_0k_base",
                "babilongv2_qa2_1k_base",
                "babilongv2_qa2_2k_base",
                "babilongv2_qa2_4k_base",
                "babilongv2_qa3_0k_base",
                "babilongv2_qa3_1k_base",
                "babilongv2_qa3_2k_base",
                "babilongv2_qa3_4k_base",
                "babilongv2_qa4_0k_base",
                "babilongv2_qa4_1k_base",
                "babilongv2_qa4_2k_base",
                "babilongv2_qa4_4k_base",
                "babilongv2_qa5_0k_base",
                "babilongv2_qa5_1k_base",
                "babilongv2_qa5_2k_base",
                "babilongv2_qa5_4k_base",
                # for debugging
                # "babilongv2_qa1_0k_base",
            ]
            metrics_result = evaluator.simple_evaluate(
                model=eval_model,
                tasks=eval_metrics,
                verbosity="WARNING",
                batch_size=self.args.per_device_eval_batch_size,
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
            dist.broadcast_object_list(metrics_to_broadcast, src=0)
            synced_lm_eval_metrics = metrics_to_broadcast[0]
            self.log(synced_lm_eval_metrics)

            return synced_lm_eval_metrics


logger = logging.getLogger(__name__)


def main():

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )

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
            # baseline
            model = AutoModelForCausalLM.from_config(
                config,
                dtype=torch_dtype,
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
        case "opt_2_rmt":
            print("opt_2_rmt")
            model = AutoModelForCausalLM.from_config(
                config,
                dtype=torch_dtype,
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
        case "opt_3_rmt":
            print("opt_3_rmt")
            # apply liger kernel
            apply_liger_kernel_to_llama()

            model = AutoModelForCausalLM.from_config(
                config,
                dtype=torch_dtype,
                attn_implementation="flash_attention_2",
            )
            model.forward = MethodType(lce_forward, model)
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
                split=None,
                shuffle=True,
                num_canonical_nodes=world_size,
                keep_zip=False,
            )

    training_args.gradient_checkpointing = False
    training_args.run_name = (
        f"{optimization_level}_batch_{training_args.per_device_train_batch_size}"
    )

    save_tokens = data_args.save_tokens
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

    trainer = PretrainRMTTrainer(
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

# liger kernel+batch 10 = 82633 tok/sec. 4gpus A100-40GB
# 82633*86_400=7_139_491_200 tok/day
# 100_000_000_000/7_139_491_200=14.00 days

# baseline rmt +batch 4 = 58004 tok/sec. 4gpus A100-40GB
# 58004*86_400=5_011_545_600 tok/day
# 100_000_000_000/5_011_545_600=19.9539 days
