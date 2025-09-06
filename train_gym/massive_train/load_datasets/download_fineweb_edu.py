import os

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from datasets import load_dataset

if __name__ == "__main__":
    load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        cache_dir="/code/fineweb_edu_10b",
    )
