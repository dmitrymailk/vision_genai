"""
украдено отсюда
- https://docs.mosaicml.com/projects/streaming/en/latest/preparing_datasets/parallel_dataset_conversion.html
- https://github.com/mosaicml/streaming/blob/main/streaming/text/convert/pile.py
- https://community.databricks.com/t5/technical-blog/managing-llm-pretraining-data-using-mosaic-data-sharding/ba-p/117158
"""

import os
import shutil
import numpy as np
import multiprocessing
from streaming import MDSWriter, StreamingDataset
from streaming.base.util import merge_index
from transformers import AutoTokenizer
from tqdm import tqdm
import math
from more_itertools import chunked

# --- 1. КОНФИГУРАЦИЯ ---

# Директория с токенизированными .npy файлами
# INPUT_DIR = "fineweb_edu_numpy_parallel"
# 02:15 на моей машине
INPUT_DIR = "fineweb_edu_10b_numpy"
# INPUT_DIR = "wikitext_2_raw_v1_numpy"
# Финальная директория для MDS датасета
# OUTPUT_DIR = "fineweb_edu_mds_chunked_padded"
# OUTPUT_DIR = os.path.abspath("fineweb_edu_10b_numpy_mds_chunked")
OUTPUT_DIR = os.path.abspath("fineweb_edu_10b_numpy_mds_chunked_1024")
# OUTPUT_DIR = os.path.abspath("wikitext_2_raw_v1_numpy_mds_chunked")

# Модель токенизатора для получения EOS токена
TOKENIZER_NAME = "unsloth/Llama-3.2-1B-Instruct"

# Параметры обработки
# CHUNK_SIZE = 2048
CHUNK_SIZE = 1024
TOKEN_DTYPE = np.uint32  # Используйте uint16, если vocab_size < 65535
# TOKEN_DTYPE = np.int64  # Используйте uint16, если vocab_size < 65535
NUM_PROC = 18  # Используем все доступные ядра

# --- 2. ФУНКЦИЯ-ВОРКЕР ДЛЯ ПАРАЛЛЕЛЬНОЙ ЗАПИСИ ---


def process_and_write_part(args):
    """
    Воркер, который читает свою часть .npy файлов, чанкует их
    и записывает в свой собственный временный MDS датасет.
    Последний неполный чанк дополняется EOS токенами.
    """
    process_idx, file_paths, temp_part_dir, tokenizer_name, chunk_size, dtype = args

    # Инициализируем токенизатор для получения ID паддинг-токена
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    pad_token_id = tokenizer.eos_token_id

    # Настройки MDSWriter с упором на скорость
    columns = {
        "input_ids": "ndarray",
        "attention_mask": "ndarray",
        "labels": "ndarray",
    }
    compression = "zstd:6"
    hashes = ["xxh64"]
    size_limit = 1 << 28  # 256MB на шард

    # Используем предварительно выделенный NumPy массив для буфера
    buffer = np.empty(chunk_size, dtype=np.int64)
    buffer_idx = 0
    samples_written = 0

    with MDSWriter(
        out=temp_part_dir,
        columns=columns,
        compression=compression,
        hashes=hashes,
        size_limit=size_limit,
    ) as out:
        for file_path in file_paths:
            tokens = np.memmap(file_path, dtype=dtype, mode="r")

            for token in tokens:
                buffer[buffer_idx] = token
                buffer_idx += 1
                if buffer_idx == chunk_size:
                    sample = {
                        "input_ids": buffer,
                        "attention_mask": np.ones_like(buffer),
                        "labels": buffer,
                    }
                    out.write(sample)
                    buffer_idx = 0
                    samples_written += 1

        # --- НОВАЯ ЛОГИКА: ОБРАБОТКА ПОСЛЕДНЕГО НЕПОЛНОГО ЧАНКА ---
        if buffer_idx > 0:
            # Заполняем оставшуюся часть буфера EOS токенами
            # buffer[buffer_idx:] = pad_token_id
            # left padding
            buffer[chunk_size - buffer_idx :] = buffer[:buffer_idx]
            buffer[: buffer_idx - 1] = pad_token_id
            labels = buffer.copy()
            mask = labels == pad_token_id
            labels[mask] = -100
            attention_mask = np.ones_like(labels)
            attention_mask[mask] = 0

            # Записываем последний, теперь уже полный, сэмпл
            sample = {
                "input_ids": buffer,
                "attention_mask": attention_mask,
                "labels": labels,
            }
            out.write(sample)
            samples_written += 1

    return samples_written


# --- 3. ОСНОВНОЙ СКРИПТ ---

if __name__ == "__main__":
    # multiprocessing.set_start_method("fork", force=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Подготовка задач
    input_dir_abs = os.path.abspath(INPUT_DIR)
    all_files = sorted(
        [
            os.path.join(input_dir_abs, f)
            for f in os.listdir(input_dir_abs)
            if f.endswith(".npy")
        ]
    )

    files_per_proc = math.ceil(len(all_files) / NUM_PROC)
    file_chunks = list(chunked(all_files, files_per_proc))

    tasks = []
    for i, chunk in enumerate(file_chunks):
        part_dir = os.path.join(OUTPUT_DIR, f"part_{i}")
        tasks.append((i, chunk, part_dir, TOKENIZER_NAME, CHUNK_SIZE, TOKEN_DTYPE))

    # Параллельная запись
    # 2 min 40 sec, 10B fineweb_edu
    print(f"Запуск {len(tasks)} параллельных процессов для записи MDS частей...")
    total_samples = 0
    with multiprocessing.Pool(processes=NUM_PROC) as pool:
        for samples_written in tqdm(
            pool.imap_unordered(process_and_write_part, tasks),
            total=len(tasks),
            desc="Обработка частей",
        ):
            total_samples += samples_written

    print(f"\nВсе части успешно записаны. Всего сэмплов: {total_samples}")

    # Логика слияния
    print("Слияние частей в единый MDS датасет...")
    merge_index(OUTPUT_DIR, keep_local=True)

    print(f"\n--- Обработка завершена! ---")
    print(f"Финальный датасет сохранен в: {OUTPUT_DIR}")

    # Верификация
    print("\n--- Проверка созданного датасета ---")
    try:
        dataset = StreamingDataset(
            remote=OUTPUT_DIR,
            local=OUTPUT_DIR,
            # split="train",
            shuffle=False,
            batch_size=64,
        )
        print(f"Датасет успешно загружен. Количество сэмплов: {len(dataset)}")
        assert len(dataset) == total_samples

        first_sample = dataset[0]
        tokens = first_sample["input_ids"]

        print(f"Форма первого сэмпла: {tokens.shape}")
        assert tokens.shape == (CHUNK_SIZE,)
        print("Проверка прошла успешно!")
    except Exception as e:
        print(f"\nПроизошла ошибка во время проверки: {e}")
