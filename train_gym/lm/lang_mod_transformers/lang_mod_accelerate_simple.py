import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    default_data_collator,
    set_seed,
)
from typing import Optional, Union
import datasets
import logging as default_logging
from dataclasses import dataclass, field
from itertools import chain
import transformers
import evaluate
from lang_mod_transformers.utils import (
    ModelArguments,
    DataTrainingArguments,
)
from tqdm.auto import tqdm
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration

logger = default_logging.getLogger(__name__)


def main():
    """Основная функция для запуска обучения"""
    # Парсим аргументы
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Настраиваем логирование
    default_logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[default_logging.StreamHandler(default_logging.sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Устанавливаем seed
    set_seed(training_args.seed)

    # Загружаем датасет
    raw_datasets = datasets.load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
    )

    # Загружаем токенизатор
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True,
    )

    # Загружаем модель с opt_1 оптимизацией
    torch_dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch_dtype,
        attn_implementation=model_args.attn_implementation,
    )

    print("model_args.attn_implementation", model_args.attn_implementation)

    # Подготавливаем датасет
    column_names = list(raw_datasets["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )

    block_size = data_args.block_size

    def group_texts(examples):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    train_dataset = lm_datasets["train"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Создаем оптимизатор
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
    )

    # Создаем dataloader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
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
    )
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    """Основной метод обучения с двумя циклами"""
    print(f"Начинаем обучение на {device}")
    print(f"Количество эпох: {training_args.num_train_epochs}")
    print(f"Размер батча: {training_args.per_device_train_batch_size}")
    print(f"Размер датасета: {len(train_dataset)}")

    # Первый цикл - по эпохам
    model.train()
    for epoch in range(int(training_args.num_train_epochs)):
        print(f"\nЭпоха {epoch + 1}/{int(training_args.num_train_epochs)}")

        total_loss = 0.0
        num_batches = 0

        # Второй цикл - по даталоадеру с tqdm
        progress_bar = tqdm(
            train_dataloader,
            desc=f"Обучение - Эпоха {epoch + 1}",
            position=0,
            leave=True,
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Обучение на одном батче

            # Перемещаем inputs на устройство
            inputs = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            loss = model(**inputs).loss

            # Backward pass
            # loss.backward()
            accelerator.backward(loss)

            # Обновляем веса
            optimizer.step()
            optimizer.zero_grad()

            # Логируем loss
            total_loss += loss.item()
            num_batches += 1

            # Обновляем progress bar
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg_loss=f"{(total_loss / num_batches):.4f}",
            )

        # Выводим средний loss за эпоху
        epoch_avg_loss = total_loss / num_batches
        print(f"Эпоха {epoch + 1} завершена. Средний loss: {epoch_avg_loss:.4f}")

    print("\nОбучение завершено!")


if __name__ == "__main__":
    main()
