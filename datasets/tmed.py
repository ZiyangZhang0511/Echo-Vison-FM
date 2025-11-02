import os
from pathlib import Path

import numpy as np
import pandas as pd
import PIL.Image as Image

import torch
from torch.utils.data import Dataset

from torchvision.transforms import v2
# import torchvision.transforms as T
from torchvision.io import read_video
from torchvision.utils import save_image


class TMED2Dataset(Dataset):
    
    def __init__(
        self,
        data_dir,
        dataset_name:str="DEV479",
        fold:str="TMED2_fold1_labeledpart",
        frames_count:int=16,
        mode:str="train",
        task:str="as_classification",
    ):
        
        self._data_dir = data_dir
        self.dataset_name = dataset_name
        self.fold = fold
        self._frames_count = frames_count
        self._mode = mode
        self._task = task
        self.label_mapping = {'no_AS': 0, 'mild_AS':1, 'mildtomod_AS': 2, 'moderate_AS':3,'severe_AS': 4}
        self.view_label_mapping = {'PLAX': 0, 'PSAX': 1, 'A4C': 2, 'A2C':3}
        self.video_list, self.diagnosis_labels_list, self.view_labels_list, self.indices_list = self._load_videos_filepath()

        if self._mode == "train":
            self.transforms = v2.Compose([
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(224, 224)),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomVerticalFlip(p=0.5),
                v2.ToDtype(torch.float32, scale=True),
            ])
        else:
            self.transforms = v2.Compose([
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(224, 224)),
                v2.ToDtype(torch.float32, scale=True),
            ])
        # if self._mode == "train":
        #     self.transforms = T.Compose([
        #         T.Lambda(lambda x: x.to(torch.uint8)/255.0),  # 模拟 ToDtype(torch.uint8, scale=True)
        #         T.Resize(size=(224, 224)),
        #         T.RandomHorizontalFlip(p=0.5),
        #         T.RandomVerticalFlip(p=0.5),
        #     #    T.Lambda(lambda x: x.to(torch.float32)/255.0),  # 模拟 ToDtype(torch.float32, scale=True)
        #     ])
        # else:
        #     self.transforms = T.Compose([
        #         T.Lambda(lambda x: x.to(torch.uint8)/255.0),  # 模拟 ToDtype(torch.uint8, scale=True)
        #         T.Resize(size=(224, 224)),
        #      #   T.Lambda(lambda x: x.to(torch.float32)/255.0),  # 模拟 ToDtype(torch.float32, scale=True)
        #     ])


    def __getitem__(self, idx):

        vframes = torch.from_numpy(self.video_list[idx].astype(np.uint8))
        # print(vframes.shape, vframes.max(), vframes.min())

        target = torch.tensor(self.diagnosis_labels_list[idx], dtype=torch.long)
        # indices = torch.tensor(self.indices_list[idx])
        num_frames = vframes.shape[0]
        indices = torch.linspace(0, num_frames-1, self._frames_count).long()
        sampled_vframes = vframes[indices]
        # print(target, indices)

        ###========get pixel values========###
        pixel_values = self.transforms(sampled_vframes)
        # print(pixel_values.shape, pixel_values.max(), pixel_values.min(), pixel_values.mean())
        # save_image(pixel_values, "./images_tmed.jpg")

        ###========get targets for different tasks========###
        if self._task == "as_classification":
            return pixel_values, target, indices
        else:
            raise ValueError(f"Argument 'task={self._task}' doesn't exist.")


    def _load_videos_filepath(self):
        # eg.~/DEV479/TMED2_fold0_labeledpart.csv
        suggested_split_csv = pd.read_csv(os.path.join(self._data_dir, f"{self.dataset_name}/{self.fold}.csv"),low_memory=False)
        
        video_list = []
        frame_indices_list = []
        diagnosis_labels_list = []
        view_labels_list = []
        selected_indices_list = []
        # 按 query_key 前缀分组 3725s1_5.png -->3721s1.avi
        grouped = suggested_split_csv.groupby(lambda x: suggested_split_csv.iloc[x]['query_key'].split('_')[0])
        
        for video_id, group in grouped:
            split = group['diagnosis_classifier_split'].iloc[0]

            if split == self._mode:
                frames = group['query_key'].tolist() # eg.3725s1_5.png
                diagnosis_label = group['diagnosis_label'].iloc[0]
                view_label = group['view_label'].iloc[0]
                
                if diagnosis_label in self.label_mapping:
                    diagnosis_label_class = self.label_mapping[diagnosis_label]
                else:
                    continue  

                # 检查视图标签是否在映射表中
                if view_label in self.view_label_mapping:
                    view_label_class = self.view_label_mapping[view_label]
                else:
                    continue  
                
                video_frames = []
                frame_indices = []  # 用于记录采样后的frame_indice

                # eg.~/view_and_diagnosis_labeled_set/labeled/labeled/1009s2_12.png
                for frame in frames:
                    file_path = os.path.join(self._data_dir, 'view_and_diagnosis_labeled_set/labeled/', frame)
                    im = self.load_image_feature(file_path)
                    video_frames.append(im)
                    frame_index = int(frame.split('_')[-1].split('.')[0])  # 提取帧序号
                    frame_indices.append(frame_index)

                # print(len(video_frames), frame_indices)
                # return
                
                # # 选择指定数量的帧
                # if len(video_frames) > self._frames_count:
                #     selected_indices = np.linspace(0, len(video_frames) - 1, self._frames_count).astype(int)
                #     video_frames = [video_frames[i] for i in selected_indices]
                #     selected_frame_indices = [frame_indices[i] for i in selected_indices]  # 记录选定帧的后半部分序号
                # else:
                #     selected_indices = np.random.choice(len(video_frames), self._frames_count, replace=True)
                #     video_frames = [video_frames[i] for i in selected_indices]
                #     selected_frame_indices = [frame_indices[i] for i in selected_indices]  # 记录选定帧的后半部分序号
                
                # 堆叠成 [F, C, H, W] 形状的张量
                video_tensor = np.stack(video_frames, axis=0)
                
                video_list.append(video_tensor)
                diagnosis_labels_list.append(diagnosis_label_class)
                view_labels_list.append(view_label_class)
                # selected_indices_list.append(selected_frame_indices)  # 记录选定的帧索引

                frame_indices_list.append(frame_indices)
                
                # print(video_tensor.shape, frame_indices)
                # return
        
        return video_list, diagnosis_labels_list, view_labels_list, frame_indices_list

    def load_image_feature(self, file_path):
        im = Image.open(file_path)
        im = np.asarray(im)
        im = im[np.newaxis, :, :].repeat(3, axis=0)  # make it (3, height, width)
        return im
    
    def __len__(self):
        return len(self.video_list)
    


if __name__ == "__main__":

    data_dir = Path("/home/olg7848/p32335/my_research/echo_vision/data/TMED2/approved_users_only")
    dataset = TMED2Dataset(data_dir)
    dataset[0]