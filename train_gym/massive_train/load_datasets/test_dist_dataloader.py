from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader


class TestDataset(Dataset):
    def __init__(self):
        self.data = list(range(401))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> int:
        return self.data[index]


def main():
    import multiprocessing

    accelerator = Accelerator()

    dataset = TestDataset()
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
    dataloader = accelerator.prepare(dataloader)

    for pos, d in enumerate(dataloader):
        print(multiprocessing.current_process().pid, pos, d)


if __name__ == "__main__":
    main()


"""
9575695757 0  0 tensor([30, 31, 32, 33, 34, 35, 36, 37, 38, 39], device='cuda:3')
tensor([20, 21, 22, 23, 24, 25, 26, 27, 28, 29], device='cuda:2')
95757 1 95756 1 tensor([70, 71, 72, 73, 74, 75, 76, 77, 78, 79], device='cuda:3')
tensor([60, 61, 62, 63, 64, 65, 66, 67, 68, 69], device='cuda:2')
95757 2 95756 2 tensor([110, 111, 112, 113, 114, 115, 116, 117, 118, 119], device='cuda:3')
tensor([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], device='cuda:2')
95757 3 95756 3 tensor([150, 151, 152, 153, 154, 155, 156, 157, 158, 159], device='cuda:3')
tensor([140, 141, 142, 143, 144, 145, 146, 147, 148, 149], device='cuda:2')
95757 4 95756 4 tensor([190, 191, 192, 193, 194, 195, 196, 197, 198, 199], device='cuda:3')
tensor([180, 181, 182, 183, 184, 185, 186, 187, 188, 189], device='cuda:2')
95757 5 95756 5 tensor([230, 231, 232, 233, 234, 235, 236, 237, 238, 239], device='cuda:3')
tensor([220, 221, 222, 223, 224, 225, 226, 227, 228, 229], device='cuda:2')
95757 6 95756 6 tensor([270, 271, 272, 273, 274, 275, 276, 277, 278, 279], device='cuda:3')
95757 7 tensor([260, 261, 262, 263, 264, 265, 266, 267, 268, 269], device='cuda:2')
95756 7 tensor([310, 311, 312, 313, 314, 315, 316, 317, 318, 319], device='cuda:3')
95757 8 tensor([300, 301, 302, 303, 304, 305, 306, 307, 308, 309], device='cuda:2')
95756 8 tensor([350, 351, 352, 353, 354, 355, 356, 357, 358, 359], device='cuda:3')
95757 9 tensor([340, 341, 342, 343, 344, 345, 346, 347, 348, 349], device='cuda:2')
95756 9 tensor([390, 391, 392, 393, 394, 395, 396, 397, 398, 399], device='cuda:3')
95757 10 tensor([380, 381, 382, 383, 384, 385, 386, 387, 388, 389], device='cuda:2')
95756 10 tensor([29, 30, 31, 32, 33, 34, 35, 36, 37, 38], device='cuda:3')
tensor([19, 20, 21, 22, 23, 24, 25, 26, 27, 28], device='cuda:2')
95755 0 tensor([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], device='cuda:1')
95755 1 tensor([50, 51, 52, 53, 54, 55, 56, 57, 58, 59], device='cuda:1')
95755 2 tensor([90, 91, 92, 93, 94, 95, 96, 97, 98, 99], device='cuda:1')
95755 3 tensor([130, 131, 132, 133, 134, 135, 136, 137, 138, 139], device='cuda:1')
95755 4 tensor([170, 171, 172, 173, 174, 175, 176, 177, 178, 179], device='cuda:1')
95755 5 tensor([210, 211, 212, 213, 214, 215, 216, 217, 218, 219], device='cuda:1')
95755 6 tensor([250, 251, 252, 253, 254, 255, 256, 257, 258, 259], device='cuda:1')
95755 7 tensor([290, 291, 292, 293, 294, 295, 296, 297, 298, 299], device='cuda:1')
95755 8 tensor([330, 331, 332, 333, 334, 335, 336, 337, 338, 339], device='cuda:1')
95755 9 tensor([370, 371, 372, 373, 374, 375, 376, 377, 378, 379], device='cuda:1')
95755 10 tensor([ 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], device='cuda:1')
95754 0 tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], device='cuda:0')
95754 1 tensor([40, 41, 42, 43, 44, 45, 46, 47, 48, 49], device='cuda:0')
95754 2 tensor([80, 81, 82, 83, 84, 85, 86, 87, 88, 89], device='cuda:0')
95754 3 tensor([120, 121, 122, 123, 124, 125, 126, 127, 128, 129], device='cuda:0')
95754 4 tensor([160, 161, 162, 163, 164, 165, 166, 167, 168, 169], device='cuda:0')
95754 5 tensor([200, 201, 202, 203, 204, 205, 206, 207, 208, 209], device='cuda:0')
95754 6 tensor([240, 241, 242, 243, 244, 245, 246, 247, 248, 249], device='cuda:0')
95754 7 tensor([280, 281, 282, 283, 284, 285, 286, 287, 288, 289], device='cuda:0')
95754 8 tensor([320, 321, 322, 323, 324, 325, 326, 327, 328, 329], device='cuda:0')
95754 9 tensor([360, 361, 362, 363, 364, 365, 366, 367, 368, 369], device='cuda:0')
95754 10 tensor([400,   0,   1,   2,   3,   4,   5,   6,   7,   8], device='cuda:0')
"""
