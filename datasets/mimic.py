from pathlib import Path
import random
import os
from tqdm.auto import tqdm

from decord import VideoReader, cpu
import numpy as np

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader, Subset

from torchvision.io import read_video
from torchvision.transforms import v2
from torchvision.utils import save_image


class MimicEchoDataset(Dataset):

    def __init__(self, data_dir:Path, data_ratio=0.25, frames_count=16, decord=True):
        
        self.data_dir = data_dir
        self.frames_count = frames_count
        self.decord = decord

        self.filespath = self._load_video_filepaths()
        filespath_length = len(self.filespath)
        # print(filespath_length)
        random.seed(42)
        self.filespath = random.sample(self.filespath, int(filespath_length*data_ratio))

        self.transforms = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            # v2.CenterCrop((224, 224)),
            v2.Resize((224, 224)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __getitem__(self, idx):

        video_filepath = self.filespath[idx]

        if self.decord:
            pixel_values, indices = self.read_frames(video_filepath)
            save_image(pixel_values, "./mimic_images.jpg")

        else:
            vframes, _, _ = read_video(str(video_filepath.resolve()), pts_unit='sec', output_format='TCHW')
            num_frames = vframes.shape[0]
            # print(vframes.size())

            indices = torch.linspace(0, num_frames-1, self.frames_count).long()

            sampled_vframes = vframes[indices]
            pixel_values = self.transforms(sampled_vframes)

        return pixel_values, indices
    
    def read_frames(self, video_path):

        vr = VideoReader(str(video_path), ctx=cpu(0))
        num_frames = len(vr)

        indices = np.linspace(0, num_frames-1, self.frames_count).astype(int)
        indices = np.clip(indices, 0, num_frames - 1)

        frames = vr.get_batch(indices)
        frames = torch.from_numpy(frames.asnumpy())
        frames = frames.permute(0, 3, 1, 2).contiguous()
        # print(frames.shape, frames.max())
        

        # save_image(frames/255, "./raw_vframes.jpg", nrow=4, padding=2, normalize=True)
        # frames[:, 0, :, :] = frames[:, 2, :, :]
        # frames[:, 1, :, :] = frames[:, 2, :, :]
        # save_image(frames/255., "./singlechannel_vframes.jpg", nrow=4, padding=2)

        
        frames = self.transforms(frames)
        # print(frames.shape, frames.max(), frames.mean())
        # save_image(frames, "./transformed_vframes.jpg", nrow=4, padding=2)

        return frames, torch.from_numpy(indices)

    def __len__(self):
        return len(self.filespath)

    def _load_video_filepaths(self):
        if isinstance(self.data_dir, list):
            filepaths = []
            for dir_path in self.data_dir:
                subfilepaths = list(dir_path.rglob("*.avi"))
                filepaths.extend(subfilepaths)
            return filepaths
        else:
            filepaths = self.data_dir.rglob("*.avi")
        # print(len(list(filepaths)))
            return list(filepaths)


class MIMICECHOIterableDataset(IterableDataset):

    def __init__(self, data_dir: Path, data_ratio=1.0, frames_count=16, shuffle=True, seed=42):
        self.data_dir = data_dir
        self.data_ratio = data_ratio
        self.frames_count = frames_count
        self.shuffle = shuffle
        self.seed = seed

        self.filespath = self._load_video_filepaths()
        filespath_length = len(self.filespath)
        # print(filespath_length)
        random.seed(self.seed)
        self.filespath = random.sample(self.filespath, int(filespath_length*data_ratio))
        # print(len(self.filespath))

        self.transforms = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.CenterCrop((224, 224)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
        ])

    def __len__(self):
        return len(self.filespath)

    def __iter__(self):

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Split data between workers
            total_workers = worker_info.num_workers
            worker_id = worker_info.id
            per_worker = len(self.filespath) // total_workers
            extra = len(self.filespath) % total_workers
            start = worker_id * per_worker + min(worker_id, extra)
            end = start + per_worker + (1 if worker_id < extra else 0)
            filepaths = self.filespath[start:end]
        else:
            filepaths = self.filespath

        if self.shuffle:
            random.seed(self.seed)
            random.shuffle(filepaths)

        # Yield one sample at a time
        for i, video_filepath in enumerate(filepaths):

            pixel_values, indices = self.read_frames(video_filepath)
            yield pixel_values, indices

            # vframes, _, _ = read_video(str(video_filepath.resolve()), pts_unit='sec', output_format='TCHW')
            # num_frames = vframes.shape[0]
            # indices = torch.linspace(0, num_frames - 1, self.frames_count).long()
            # sampled_vframes = vframes[indices]
            # pixel_values = self.transforms(sampled_vframes)
            # yield pixel_values, indices

    def read_frames(self, video_path):
        vr = VideoReader(str(video_path), ctx=cpu(0))
        num_frames = len(vr)

        indices = np.linspace(0, num_frames - 1, self.frames_count).astype(int)
        indices = np.clip(indices, 0, num_frames - 1)

        frames = vr.get_batch(indices)
        frames = torch.from_numpy(frames.asnumpy())
        frames = frames.permute(0, 3, 1, 2)
        
        frames = self.transforms(frames)

        return frames, torch.from_numpy(indices)


    def _load_video_filepaths(self):
        filepaths = self.data_dir.rglob("*.avi")
        # print(len(list(filepaths)))
        return list(filepaths)


if __name__ == "__main__":
    # pass

    # data_dir = Path("/home/olg7848/p32335/MIMIC-ECHO/mimic_echo_avi")
    # dataset = MIMICECHOIterableDataset(data_dir=data_dir, data_ratio=0.25, frames_count=16, shuffle=False)

    # dataloader = DataLoader(dataset, batch_size=64, num_workers=8)
    # print(len(dataloader.dataset.filespath))

    # for epoch in range(2):
    #     for batch in tqdm(dataloader, total=len(dataloader.dataset.filespath) // 64):
    #         pixel_values, indices = batch
    #     print(pixel_values.shape, indices.shape)



    # data_dir = Path("/home/olg7848/p32335/MIMIC-ECHO/mimic_echo_avi/p14")
    # dataset = MimicEchoDataset(data_dir=data_dir, data_ratio=0.25, frames_count=16)
    # dataset_length = len(dataset)
    # for _ in dataset:
    #     break
    
    # dataloader = DataLoader(dataset, batch_size=64, num_workers=0)
    # for batch in tqdm(dataloader):
    #     pixel_values, indices = batch
    #     print(pixel_values.size(), indices.size())
    #     print(indices[0])
    #     break

    # data_dir = [
    #     Path("/home/olg7848/p32335/MIMIC-ECHO/mimic_echo_avi/p11"),
    #     Path("/home/olg7848/p32335/MIMIC-ECHO/mimic_echo_avi/p13"),
    #     Path("/home/olg7848/p32335/MIMIC-ECHO/mimic_echo_avi/p14"),
    #     Path("/home/olg7848/p32335/MIMIC-ECHO/mimic_echo_avi/p15"),
    # ]

    # dataset = MimicEchoDataset(data_dir=data_dir, data_ratio=1.0, frames_count=16)
    # print(len(dataset))
    # dataset[10]

    video_path = Path("/home/olg7848/p32335/MIMIC-ECHO/mimic_echo_avi/p16/p16_p16002684_94221748_0006.avi")
    vr = VideoReader(str(video_path), ctx=cpu(0))
    num_frames = len(vr)
    print(num_frames)

    indices = np.linspace(0, num_frames-1, 16).astype(int)
    indices = np.clip(indices, 0, num_frames - 1)
    print(indices)

    frames = vr.get_batch(indices)
    frames = torch.from_numpy(frames.asnumpy())
    frames = frames.permute(0, 3, 1, 2).contiguous()
    print(frames.shape, frames.max())

    save_image(frames/255., "./frames.jpg", nrow=4, padding=2)