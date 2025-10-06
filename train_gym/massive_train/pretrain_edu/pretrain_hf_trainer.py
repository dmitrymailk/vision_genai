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

    def _evaluate(self, *args, **kwargs):
        self.model.eval()
        train_metadata = getattr(self.state, "train_metadata", None)
        # если метадата None и global_step ноль, значит это самая первая эвалюация модели
        # или если метадата не None и глобальный шаг больше нуля, значит что мы продложили тренировку и
        # это не самый первый шаг при продложении
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
                "babilongv2_qa1_under_4k_base",
                "babilongv2_qa2_under_4k_base",
                "babilongv2_qa3_under_4k_base",
                "babilongv2_qa4_under_4k_base",
                "babilongv2_qa5_under_4k_base",
            ]
            metrics_result = evaluator.simple_evaluate(
                model=eval_model,
                tasks=target_metrics,
                verbosity="WARNING",
                batch_size=self.args.per_device_eval_batch_size,
            )
            gc.collect()
            torch.cuda.empty_cache()
            self.accelerator.wait_for_everyone()
            report_dict = {}
            if self.accelerator.is_main_process:
                metrics_result = metrics_result["results"]
                # print(metrics_result)
                ban_keys = ["stderr", "alias"]
                for metric_name in target_metrics:
                    for key, value in metrics_result[metric_name].items():
                        eval_key = f"eval_{metric_name}_{key}"
                        if not any(ban_key in eval_key for ban_key in ban_keys):
                            report_dict[eval_key] = value
                # в общем wandb иногда багает и неравильно отображает шаги относительно метрик
                # это в целом можно решить в графическом интерфейсе, но на всякий можно явно указывать
                # в логах какой это шаг
                # self.accelerator.log(
                #     report_dict,
                #     step=self.state.global_step,
                # )

            # данный объект используется например для отбора лучшего чекпоинта, поэтому его нужно передать дальше
            # по всем потокам
            metrics_to_broadcast = [report_dict]
            dist.broadcast_object_list(metrics_to_broadcast, src=0)
            synced_lm_eval_metrics = metrics_to_broadcast[0]
            self.log(synced_lm_eval_metrics)

            return synced_lm_eval_metrics


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


@dataclass
class DataTrainingArguments:

    dataloader_type: Optional[str] = field(
        default="hf_edu",
        metadata={
            "help": "The type dataloader_type (default hf method or mosaic streaming)"
        },
    )
    max_train_tokens: Optional[int] = field(
        default=405_635_072,
        metadata={"help": ("Max train tokens")},
    )
    save_tokens: Optional[int] = field(
        default=400_000_000,
        metadata={"help": ("save model after seen tokens")},
    )
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
        case "opt_1":
            print("opt_1")
            model = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
            )
        case "opt_2":
            print("opt_2")
            apply_liger_kernel_to_llama()
            model = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=torch_dtype,
                attn_implementation="flash_attention_2",
            )

        case "opt_3":
            print("opt_3")
            model = AutoModelForCausalLM.from_config(
                config,
                torch_dtype=torch_dtype,
                attn_implementation=model_args.attn_implementation,
            )

            # unsloth version
            # данный метод работает если reduction cross entropy mean
            # и у нас нет никаких accumulation steps. однако если у нас pretrain
            # все метки идут плотно и поэтому можно в целом складывать средние лоссы
            # но в общем случае нам нужно делить loss на количество токенов которые
            # породили этот лосс. иными словами reduction mean подходит не для всех случаев
            # только когда у нас одинаковое количество токенов во ВСЕХ батчах
            # https://huggingface.co/blog/gradient_accumulation
            # https://unsloth.ai/blog/gradient
            model.forward = MethodType(cce_forward, model)

            major, minor = torch.cuda.get_device_capability()
            # Target version
            target_major = 8
            target_minor = 9
            if (major > target_major) or (
                major == target_major and minor >= target_minor
            ):
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

            for m in reversed(list(model.modules())):
                if isinstance(m, LlamaDecoderLayer):
                    m.compile(
                        backend="inductor",
                        # mode="max-autotune",
                    )

    match dataloader_type:
        case "hf_edu":
            # TODO: это неправильная отладочная версия которая выкидывает данные
            # ее нужно переписать на правильную или вообще убрать на обычную загрузку
            # претокенизированного датасета
            with training_args.main_process_first(
                desc="dataset loading and processing"
            ):
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
                    num_proc=min(os.cpu_count() - 1, 64),
                    remove_columns=column_names,
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
                    num_proc=min(os.cpu_count(), 64),
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
    # save_tokens = 400_000_000
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
