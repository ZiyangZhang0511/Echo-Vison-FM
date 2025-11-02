import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from torchvision.transforms import v2
from torchvision.io import read_video
from torchvision.utils import save_image

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class EchonetDynamicDataset(Dataset):
    
    def __init__(self, data_dir, data_ratio=1.0, frames_count:int=16, mode:str="train", task:str="ef_classification"):
        
        self._data_dir = data_dir
        self._frames_count = frames_count
        self._mode = mode
        self._task = task

        self.filespath = self._load_videos_filepath()
        num_files = len(self.filespath)

        if self._mode == "train":
            self.filespath = random.sample(self.filespath, int(num_files*data_ratio))
        
        # print(num_files, len(self.filespath))



        self.annotation = pd.read_csv(Path(self._data_dir/'FileList.csv'))
        
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

    def __getitem__(self, idx):

        video_filepath = self.filespath[idx]

        ###========get pixel values========###
        vframes, _, _ = read_video(str(video_filepath.resolve()), pts_unit='sec', output_format='TCHW')
        num_frames = vframes.shape[0]
        indices = torch.linspace(0, num_frames-1, self._frames_count).long()
        sampled_vframes = vframes[indices]
        pixel_values = self.transforms(sampled_vframes)

        ###========get targets for different tasks========###
        row = self.annotation[self.annotation["FileName"] == video_filepath.stem]

        if self._task == "ef_regression":
            return pixel_values, torch.tensor([row["EF"].item()/100.0]), indices
        elif self._task == "esv_regression":
            return pixel_values, torch.tensor([row["ESV"].item()]).log(), indices
        elif self._task == "edv_regression":
            return pixel_values, torch.tensor([row["EDV"].item()]).log(), indices
        elif self._task == "ef_classification":
            ef_value = torch.tensor([row["EF"].item()/100.0])
            ef_label = 0. if ef_value < 0.50 else 1.
            return pixel_values, torch.tensor(ef_label).float(), indices
        else:
            raise ValueError(f"Argument 'task={self._task}' doesn't exist.")


    def _load_videos_filepath(self):
        files_df = pd.read_csv(Path(self._data_dir)/'FileList.csv')

        files_df['Split'] = files_df['Split'].str.lower()
        split_data = files_df[files_df['Split'] == self._mode]

        videos_filepath = []
        for _, row in split_data.iterrows():
            video_filepath = Path(self._data_dir/'Videos'/(row["FileName"]+'.avi'))
            videos_filepath.append(video_filepath)

        return videos_filepath

    def __len__(self):
        return len(self.filespath)


# import torch
# from torch.utils.data import Dataset

# # from torchvision.transforms import v2
# import torchvision.transforms as T
# from torchvision.io import read_video
# import numpy as np
# import pandas as pd
# import os
# from pathlib import Path
# from sklearn.preprocessing import MinMaxScaler

class EchonetLVHDataset(Dataset):
    
    def __init__(self, data_dir, frames_count:int=16, mode:str="train", task:str="IVS_regression"):
        
        self._data_dir = data_dir
        self._frames_count = frames_count
        self._mode = mode
        self._task = task

        self.annotation = pd.read_csv(Path(self._data_dir/'MeasurementsList.csv'))
        self.clean_data = self._load_cleandata(self.annotation)
        self.filespath = self._load_videos_filepath()

        # print(len(self.clean_data), self.clean_data.head(4))
        # print(self.filespath[:2])

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

    def __getitem__(self, idx):

        video_filepath = self.filespath[idx]
        vframes, _, _ = read_video(str(video_filepath.resolve()), pts_unit='sec', output_format='TCHW')
        print(vframes.max())
        num_frames = vframes.shape[0]
        indices = torch.linspace(0, num_frames-1, self._frames_count).long()
        sampled_vframes = vframes[indices]
        pixel_values = self.transforms(sampled_vframes)
        # print(vframes.shape)

        # save_image(pixel_values, "./image.jpg")



        ###========get pixel values========###
        Calc_file = self._task.split('_')[0]+'d'
        row = self.clean_data[(self.clean_data["HashedFileName"] == video_filepath.stem) & (self.clean_data["Calc"]==Calc_file)]
        print(row)
        ###========get targets for different tasks========###

        target = torch.tensor(row[["X1", "Y1", "X2", "Y2"]].values.flatten(), dtype=torch.float32)
        if self._task == "IVS_regression":
          #  print(pixel_values.shape,target.shape,indices.shape)
            return pixel_values, target, indices
        elif self._task == "LVPW_regression":
            return pixel_values, target, indices
        elif self._task == "LVID_regression":
            return pixel_values, target, indices
        
        else:
            raise ValueError(f"Argument 'task={self._task}' doesn't exist.")

    def _load_cleandata(self, data):
        data["HashedFileName"].value_counts()
        sub_mask = data["HashedFileName"].value_counts() >= 3
        mask = data["HashedFileName"].isin(data["HashedFileName"].value_counts()[sub_mask].index)
        clean_data = data[mask]

        IVSd  = "IVSd"
        LVIDd = "LVIDd"
        LVPWd = "LVPWd"

        calc_data  = clean_data.groupby(by="HashedFileName")[["Calc"]].value_counts().to_frame().query("Calc == @IVSd or Calc == @LVIDd or Calc == @LVPWd")
        calc_mask  = calc_data.groupby(level=[0]).size() == 3
        calc_mask  = calc_mask[calc_mask == True]
        clean_data = clean_data[clean_data["HashedFileName"].isin(calc_mask.index)]
        # Optional to only take diastoles
        data_mask = (clean_data["Calc"] == IVSd) | (clean_data["Calc"] == LVIDd) | (clean_data["Calc"] == LVPWd)
        clean_data = clean_data[data_mask]
        clean_data = clean_data[~clean_data.duplicated(["HashedFileName", "Calc"])]
        clean_data.reset_index(inplace=True, drop=True)

        # 归一化列 ["X1", "Y1", "X2", "Y2"]
        columns_to_normalize = ["X1", "Y1", "X2", "Y2"]
        scaler = MinMaxScaler()
        # 打印每个列的最小值和最大值
        for column in columns_to_normalize:
            print(f"{column} min: {clean_data[column].min()}, max: {clean_data[column].max()}")
        # 计算最小值和最大值，并进行归一化
        clean_data[columns_to_normalize] = scaler.fit_transform(clean_data[columns_to_normalize])
        return clean_data
    
    def _load_videos_filepath(self):
        files_df = self.clean_data

        files_df['split'] = files_df['split'].str.lower()
        split_data = files_df[(files_df['split'] == self._mode) & (files_df["Calc"].str.contains(self._task.split('_')[0]))]
        videos_filepath = []
        for _, row in split_data.iterrows():
            fileName = row["HashedFileName"] + '.avi'
            video_filepath = self._find_file_path(self._data_dir,fileName)
            if video_filepath is not None:
                videos_filepath.append(video_filepath)
            else:
                print(f"{fileName} is not found!")
        return videos_filepath

    def _find_file_path(self,folder_path, filename):
        for root, dirs, files in os.walk(folder_path):
            if filename in files:
                # 构建完整的文件路径
                full_path = Path(os.path.join(root, filename))
                return full_path
        return None
    
    def __len__(self):
        return len(self.filespath)




if __name__ == "__main__":

    data_dir = Path("/home/olg7848/p32335/my_research/echo_vision/data/echonet_dynamic")

    dataset = EchonetDynamicDataset(data_dir, data_ratio=0.1, mode="train")

    # re = dataset[0]
    # print(re[0].shape, re[1])