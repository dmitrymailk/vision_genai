import os
import multiprocessing
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# --- 1. КОНФИГУРАЦИЯ ---

# Параметры датасета
DATASET_NAME = "HuggingFaceFW/fineweb-edu"
# 11:54 my server 4090
DATASET_CONFIG = "sample-10BT"
SPLIT = "train"
CACHE_DIR = "fineweb_edu_10b"

# Модель токенизатора (ВАЖНО: замените на ваш)
TOKENIZER_NAME = "unsloth/Llama-3.2-1B-Instruct"

# Параметры обработки
NUM_PROC = 17  # Количество параллельных процессов
TOKENIZATION_BATCH = 10_000  # Размер батча для токенизации

# Директория для сохранения результатов
# OUTPUT_DIR = "fineweb_edu_10b_numpy"
OUTPUT_DIR = "wikitext_2_raw_v1_numpy"


# --- 2. ФУНКЦИЯ-ВОРКЕР ДЛЯ ПАРАЛЛЕЛЬНОЙ ОБРАБОТКИ ---


def process_shard(args):
    """
    Обрабатывает один шард датасета в точности по логике,
    предоставленной в исходном коде.
    """
    # Распаковываем аргументы
    rank, shard, tokenizer_name, tokenization_batch, output_dir = args

    # Инициализируем токенизатор внутри процесса
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # Получаем ID специальных токенов
    BOS_TOKEN_ID = [tokenizer.bos_token_id]
    EOS_TOKEN_ID = [tokenizer.eos_token_id]

    batch = []
    batch_num_counter = 0

    # Обертка tqdm для отслеживания прогресса внутри одного процесса
    progress_bar = tqdm(
        total=len(shard),
        desc=f"Процесс {rank}",
        position=rank,  # Для красивого отображения нескольких прогресс-баров
        leave=False,
    )

    for idx, item in enumerate(shard):
        batch.append(item["text"])

        # Когда батч набрался, обрабатываем его
        if len(batch) >= tokenization_batch:
            # Выполняем токенизацию
            tok_result = tokenizer(
                batch,
                truncation=False,
                padding=False,
                add_special_tokens=False,
            )["input_ids"]

            # Добавляем BOS/EOS и объединяем все в один большой список
            flattened_tokens = []
            for token_ids in tok_result:
                flattened_tokens.extend(BOS_TOKEN_ID + token_ids + EOS_TOKEN_ID)

            # Сохраняем батч на диск
            np_array = np.array(
                flattened_tokens,
                dtype=np.uint32,
            )

            # Создаем уникальное имя файла
            filename = f"shard_{rank:02d}_batch_{batch_num_counter:04d}.npy"
            filepath = os.path.join(output_dir, filename)

            # Используем np.memmap для записи
            fp = np.memmap(
                filepath,
                dtype=np.uint32,
                mode="w+",
                shape=np_array.shape,
            )
            fp[:] = np_array[:]
            fp.flush()

            batch_num_counter += 1

            # Сбрасываем батч
            batch = []

        progress_bar.update(1)

    # Обработка последнего батча (если он неполный)
    if len(batch) > 0:
        tok_result = tokenizer(
            batch,
            truncation=False,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]

        flattened_tokens = []
        for token_ids in tok_result:
            flattened_tokens.extend(BOS_TOKEN_ID + token_ids + EOS_TOKEN_ID)

        if flattened_tokens:
            np_array = np.array(flattened_tokens, dtype=np.uint32)
            filename = f"shard_{rank:02d}_batch_{batch_num_counter:04d}.npy"
            filepath = os.path.join(output_dir, filename)

            fp = np.memmap(
                filepath,
                dtype=np.uint32,
                mode="w+",
                shape=np_array.shape,
            )
            fp[:] = np_array[:]
            fp.flush()

    progress_bar.close()
    return f"Процесс {rank} завершен."


if __name__ == "__main__":
    # Метод 'fork' часто лучше работает с библиотеками типа transformers/tokenizers
    # multiprocessing.set_start_method("fork", force=True)

    # --- 3. ПОДГОТОВКА ДАННЫХ ---
    print("Загрузка датасета...")
    # dataset = load_dataset(
    #     DATASET_NAME,
    #     name=DATASET_CONFIG,
    #     split=SPLIT,
    #     cache_dir=CACHE_DIR,
    #     num_proc=NUM_PROC,
    # )
    dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
    )
    dataset = dataset["train"]
    dataset = dataset.remove_columns(
        column_names=[item for item in dataset.features.keys() if item != "text"]
    )

    # Используем 1/6 часть датасета
    # dataset = dataset.select(range(len(dataset) // 6))
    print(f"Размер датасета для обработки: {len(dataset)} документов")

    # Создаем выходную директорию
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Результаты будут сохранены в: {OUTPUT_DIR}")

    print("Разделение датасета на шарды...")
    shards = [
        dataset.shard(
            num_shards=NUM_PROC,
            index=rank,
            contiguous=True,
        )
        for rank in range(NUM_PROC)
    ]

    # --- 4. ЗАПУСК ПАРАЛЛЕЛЬНОЙ ОБРАБОТКИ ---

    # Подготовка аргументов для каждого процесса
    tasks = [
        (i, shards[i], TOKENIZER_NAME, TOKENIZATION_BATCH, OUTPUT_DIR)
        for i in range(NUM_PROC)
    ]

    print(f"Запуск {NUM_PROC} параллельных процессов...")
    with multiprocessing.Pool(NUM_PROC) as pool:
        # pool.imap_unordered вернет результаты по мере их готовности
        results = list(pool.imap_unordered(process_shard, tasks))

    for res in results:
        print(res)

    # --- 5. ВЫВОД РЕЗУЛЬТАТОВ ---
    files_created = len(os.listdir(OUTPUT_DIR))
    print("\n--- ОБРАБОТКА ЗАВЕРШЕНА ---")
    print(f"Создано {files_created} файлов в директории '{OUTPUT_DIR}'")
