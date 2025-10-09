from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader


class TestDataset(Dataset):
    def __init__(self):
        self.data = list(range(401))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> int:
        return self.data[index]


from accelerate.utils import DataLoaderConfiguration
from transformers.trainer_pt_utils import AcceleratorConfig
from accelerate.utils import send_to_device
from streaming import MDSWriter, StreamingDataset
from torch.utils.data import DataLoader
from transformers import default_data_collator
import torch

import os


class DeviceDataLoader(DataLoader):
    def __iter__(self):
        cpu_iterator = super().__iter__()
        current_device = torch.cuda.current_device()
        current_device = torch.device(f"cuda:{current_device}")
        for batch in cpu_iterator:
            yield send_to_device(batch, current_device)


def main():
    import multiprocessing

    accelerator_config = AcceleratorConfig(
        dispatch_batches=False,
    )
    dataloader_params = [
        "split_batches",
        "dispatch_batches",
        "even_batches",
        "use_seedable_sampler",
    ]

    dataloader_config = DataLoaderConfiguration(
        **{param: accelerator_config.pop(param) for param in dataloader_params}
    )

    accelerator = Accelerator(
        dataloader_config=dataloader_config,
    )

    # local_dir = "fineweb_edu_10b_numpy_mds_chunked"
    # local_dir = "fineweb_edu_10b_numpy_mds_chunked_1024"
    local_dir = "fineweb_edu_10b_numpy_mds_chunked_2048"
    # local_dir = "fineweb_edu_10b_numpy_mds_chunked"
    batch_size = 10
    # batch_size = 16
    # batch_size = 4
    dataset = StreamingDataset(
        local=local_dir,
        remote=local_dir,
        batch_size=batch_size,
        # batch_size=1,
        # batch_size=64,
        split=None,
        shuffle=True,
    )

    dataloader = DeviceDataLoader(
        dataset,
        batch_size=batch_size,
        # pin_memory=True,
        num_workers=4,
        collate_fn=default_data_collator,
        # collate_fn=data_collator_streaming_fix,
        drop_last=True,
        # shuffle=True,
        # persistent_workers=True,
    )
    # не нужно
    # dataloader = accelerator.prepare(dataloader)

    for pos, d in enumerate(dataloader):
        # inputs = accelerator.prepare(d["input_ids"][:10])
        inputs = d["input_ids"][:10]
        result = f"{len(dataloader)}_{inputs}_{inputs.device}"
        print(result)
        break


if __name__ == "__main__":
    main()


"""
118336_tensor([[  311,  4048,   872,  ...,  1115,  1436,   387],
        [  320,  3692, 43346,  ...,  9367,    16,   274],
        [58641,  1053,  2997,  ...,   872,  4868,   323],
        ...,
        [  505,   279,   432,  ..., 57811, 37027,  3445],
        [ 2466,   323,  2678,  ...,   374,   279,  1888],
        [  323,  1587,   539,  ...,  9951,  5149,   311]], device='cuda:2')_cuda:2
118336_tensor([[  279,  3925,  6603,  ...,    12,   549,   652],
        [  459,  2926, 15489,  ...,   956,  1120,   733],
        [19338,   627,   644,  ..., 16828,  9064,   320],
        ...,
        [  498,   323, 63024,  ...,    20,   430,   279],
        [  291, 11509,   477,  ..., 11983,   304,  1778],
        [  432,  6048,    69,  ...,    13,    15, 87401]], device='cuda:1')_cuda:1
118336_tensor([[22632,   311,   387,  ...,  3723,  4273,   304],
        [ 2748, 19646,    13,  ...,   584,  2011,  3009],
        [ 1766, 24923,   279,  ..., 17357,    11,  2911],
        ...,
        [  389, 25031, 19207,  ...,    13, 43474,  1521],
        [ 9843,   279, 12434,  ...,  5663, 34465,     8],
        [ 2759,    11, 16450,  ...,   311,   279,  1023]], device='cuda:3')_cuda:3
118336_tensor([[  315, 35113, 59904,  ..., 12973,    13,   763],
        [ 1054,  6219,   358,  ..., 36330,   220,  1758],
        [  264,  2763,   810,  ...,  1405,   358,  1390],
        ...,
        [ 1047,  3549,  1778,  ...,   315, 11165,   268],
        [ 2349,  3445,  2349,  ...,  2403, 72915, 15853],
        [   11,  1054,  8747,  ...,   304,  1274,  3515]], device='cuda:0')_cuda:0
"""
