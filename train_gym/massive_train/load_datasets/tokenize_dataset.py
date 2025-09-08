import os
import multiprocessing
from tqdm.auto import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset  # Предполагается, что dataset уже загружен
from transformers import AutoTokenizer
import numpy as np

# --- Глобальные параметры ---
tokenizer = AutoTokenizer.from_pretrained("unsloth/Llama-3.2-1B-Instruct")
MAX_LENGTH = 2048
BOS_TOKEN_ID = [tokenizer.bos_token_id]
EOS_TOKEN_ID = [tokenizer.eos_token_id]
WRITE_BATCH_SIZE = 10_000
ITER_BATCH_SIZE = 1000
OUTPUT_DIR = "fineweb_edu_chunked_sharded_parquet_numpy"
NUM_PROC = 16  # Устанавливаем количество процессов, равное числу шардов


def process_shard(shard_info):
    """
    Обрабатывает один шард датасета и записывает его в отдельный Parquet-файл.
    """
    shard_index, total_shards, dataset_obj = shard_info

    # Получаем наш уникальный шард
    # `shard_index` - от 0 до NUM_PROC-1
    # `total_shards` - NUM_PROC
    # `contiguous=True` - важно для производительности
    shard = dataset_obj.shard(
        num_shards=total_shards, index=shard_index, contiguous=True
    )

    output_file = os.path.join(OUTPUT_DIR, f"part-{shard_index:05d}.parquet")

    # --- Логика из предыдущего однопоточного решения ---
    token_buffer = []
    chunk_batch_buffer = []
    writer = None

    try:
        num_batches = len(shard) // ITER_BATCH_SIZE + 1

        # Используем desc, чтобы видеть, какой воркер что делает
        progress_bar = tqdm(
            shard.iter(batch_size=ITER_BATCH_SIZE),
            total=num_batches,
            desc=f"Shard {shard_index}/{total_shards}",
            position=shard_index,  # Чтобы прогресс-бары не накладывались друг на друга
        )

        for batch in progress_bar:
            for iids in batch["input_ids"]:
                token_buffer.extend(BOS_TOKEN_ID)
                token_buffer.extend(iids)
                token_buffer.extend(EOS_TOKEN_ID)

            while len(token_buffer) >= MAX_LENGTH:
                # chunk = token_buffer[:MAX_LENGTH]
                chunk = np.array(
                    token_buffer[:MAX_LENGTH],
                    dtype=np.uint32,
                )  # .reshape(1, MAX_LENGTH)
                chunk_batch_buffer.append({"input_ids": chunk})
                token_buffer = token_buffer[MAX_LENGTH:]

                if len(chunk_batch_buffer) >= WRITE_BATCH_SIZE:
                    table = pa.Table.from_pylist(chunk_batch_buffer)
                    if writer is None:
                        writer = pq.ParquetWriter(
                            output_file, table.schema, compression="snappy"
                        )
                    writer.write_table(table)
                    chunk_batch_buffer = []

        if chunk_batch_buffer:
            table = pa.Table.from_pylist(chunk_batch_buffer)
            if writer is None:
                writer = pq.ParquetWriter(
                    output_file, table.schema, compression="snappy"
                )
            writer.write_table(table)

    finally:
        if writer:
            writer.close()

    return f"Shard {shard_index} finished, wrote to {output_file}"


if __name__ == "__main__":

    def tokenization(example):
        return tokenizer(
            example["text"],
            truncation=False,
            padding=False,
            add_special_tokens=False,
        )

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        cache_dir="fineweb_edu_10b",
    )
    dataset = dataset.select(list(range(len(dataset) // 3)))
    dataset = dataset.map(
        tokenization,
        batched=True,
        num_proc=18,
    )
    dataset = dataset.remove_columns(
        column_names=[item for item in dataset.features.keys() if item != "input_ids"]
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Создаем список задач для пула процессов
    # Передаем сам объект датасета в каждый воркер.
    # Благодаря Arrow, это делается эффективно (zero-copy)
    tasks = [(i, NUM_PROC, dataset) for i in range(NUM_PROC)]

    print(f"Starting processing on {NUM_PROC} cores...")

    # Создаем и запускаем пул процессов
    # 'fork' - важно для *nix систем, чтобы избежать повторной загрузки данных
    # context = multiprocessing.get_context('fork')
    with multiprocessing.Pool(processes=NUM_PROC) as pool:
        # map применяет функцию process_shard к каждому элементу из `tasks`
        for result in tqdm(
            pool.imap_unordered(process_shard, tasks),
            total=len(tasks),
            desc="Total Progress",
        ):
            print(result)

    print("\nAll shards processed successfully.")
    print(f"Output is a Parquet dataset directory: '{OUTPUT_DIR}'")

    # --- 3. Как использовать результат ---
    from datasets import load_dataset

    print("\nLoading the sharded dataset back...")
    # Просто указываем путь к папке!
    final_dataset = load_dataset("parquet", data_dir=OUTPUT_DIR)

    print("Dataset loaded successfully:")
    print(final_dataset)
    print("\nFirst sample:")
    print(final_dataset["train"][0])
