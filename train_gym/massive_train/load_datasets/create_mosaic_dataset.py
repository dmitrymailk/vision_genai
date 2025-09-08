"""

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
INPUT_DIR = "fineweb_edu_numpy_parallel"
# Финальная директория для MDS датасета
# OUTPUT_DIR = "fineweb_edu_mds_chunked_padded"
OUTPUT_DIR = os.path.abspath("fineweb_edu_mds_chunked")

# Модель токенизатора для получения EOS токена
TOKENIZER_NAME = "unsloth/Llama-3.2-1B-Instruct"

# Параметры обработки
CHUNK_SIZE = 2048
TOKEN_DTYPE = np.uint32  # Используйте uint16, если vocab_size < 65535
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
    columns = {"tokens": "ndarray"}
    compression = "zstd:6"
    hashes = ["xxh64"]
    size_limit = 1 << 28  # 256MB на шард

    # Используем предварительно выделенный NumPy массив для буфера
    buffer = np.empty(chunk_size, dtype=dtype)
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
                    sample = {"tokens": buffer}
                    out.write(sample)
                    buffer_idx = 0
                    samples_written += 1

        # --- НОВАЯ ЛОГИКА: ОБРАБОТКА ПОСЛЕДНЕГО НЕПОЛНОГО ЧАНКА ---
        if buffer_idx > 0:
            # print(
            #     f"Процесс {process_idx}: найден остаток из {buffer_idx} токенов. Дополняем до {chunk_size}..."
            # )
            # Заполняем оставшуюся часть буфера EOS токенами
            buffer[buffer_idx:] = pad_token_id

            # Записываем последний, теперь уже полный, сэмпл
            sample = {"tokens": buffer}
            out.write(sample)
            samples_written += 1

    return samples_written


# --- 3. ОСНОВНОЙ СКРИПТ ---

if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)

    if not os.path.isdir(INPUT_DIR):
        raise FileNotFoundError(f"Входная директория не найдена: '{INPUT_DIR}'")

    if os.path.exists(OUTPUT_DIR):
        print(f"Удаление существующей директории: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    # TEMP_DIR будет создан внутри OUTPUT_DIR
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
        # TEMP_DIR уже абсолютный, поэтому part_dir тоже будет абсолютным
        part_dir = os.path.join(OUTPUT_DIR, f"part_{i}")
        tasks.append((i, chunk, part_dir, TOKENIZER_NAME, CHUNK_SIZE, TOKEN_DTYPE))

    # Параллельная запись
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
        tokens = first_sample["tokens"]

        print(f"Форма первого сэмпла: {tokens.shape}")
        assert tokens.shape == (CHUNK_SIZE,)
        print("Проверка прошла успешно!")
    except Exception as e:
        print(f"\nПроизошла ошибка во время проверки: {e}")
