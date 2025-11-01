import os
import configparser
from io import StringIO
from pathlib import Path

import numpy as np
import nibabel as nib

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from torchvision.utils import save_image

from .utils import *


class CamusDataset(Dataset):
    def __init__(self, data_dir:Path, frames_count:int=16, mode:str="train", task:str="ef_classification"):

        self.data_dir = data_dir
        self.frames_count = frames_count
        self.mode = mode
        self.task = task

        self.patients_id = self.generate_patients_id()[self.mode]
        # self.patients_id = self.get_data_split()

        self.suffix = "_4CH_half_sequence.nii.gz"
        self.annotation_filename = "Info_4CH.cfg"

        if self.mode == "train":
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

        ###========get video path and its annotation file path========###
        patient_id = self.patients_id[idx]
        video_filepath, annotation_filepath = self._get_video_filepath(patient_id)

        seq, seq_gt, frame_pairs_mask, ef, edv, esv, spacing = self.preprocess_data(patient_id, resize_size=(224, 224))
        # print(seq.shape, seq.max(), seq.min(), ef, edv, esv)

        ###========get pixel values========###
        # vframes = torch.tensor(nib.load(video_filepath).get_fdata())
        # vframes = vframes.permute(2, 0, 1).unsqueeze(1).repeat(1, 3, 1, 1)
        # num_frames = vframes.shape[0]
        # # print(num_frames)
        # indices = torch.linspace(0, num_frames-1, self.frames_count).long()
        # sampled_vframes = vframes[indices]

        num_frames = seq.shape[0]
        indices = torch.linspace(0, num_frames-1, self.frames_count).long()
        sampled_vframes = torch.from_numpy(seq[indices].astype(np.uint8))
        # print(sampled_vframes.max(), sampled_vframes.min())
        pixel_values = self.transforms(sampled_vframes)
        # print(pixel_values.max(), pixel_values.min(), pixel_values.mean())

        # save_image(pixel_values, "./image_camus.jpg")

        ###========get targets for different tasks========###
        # config = self._read_cfg_file(annotation_filepath)
        if self.task == "ef_regression":
            return pixel_values, torch.tensor(ef/100.).float(), indices
        elif self.task == "esv_regression":
            return pixel_values, torch.tensor(esv/100.).float(), indices
        elif self.task == "edv_regression":
            return pixel_values, torch.tensor(edv/100.).float(), indices
        elif self.task == "ef_classification":
            ef_value = torch.tensor(ef/100.)
            ef_label = 0. if ef_value < 0.45 else 1.
            return pixel_values, torch.tensor(ef_label).float(), indices
        else:
            raise ValueError(f"Argument 'task={self.task}' doesn't exist.")

    def __len__(self):
        return len(self.patients_id)

    def preprocess_data(self, patient_name, resize_size):
        patient_root = self.data_dir / "database_nifti"
        # patient_root = self.data_dir
        patient_dir = patient_root / patient_name

        seq_pattern = "{patient_name}_{view}_half_sequence.nii.gz"
        seq_gt_pattern = "{patient_name}_{view}_half_sequence_gt.nii.gz"
        cfg_pattern = "Info_{view}.cfg"
        
        ef, edv, esv = self.compute_ef_to_patient(patient_dir=patient_dir)

        for view in ["2CH","4CH"]:
        # for view in ["4CH", "2CH"]:
            seq_name = seq_pattern.format(patient_name=patient_name, view=view)
            seq_gt_name = seq_gt_pattern.format(patient_name=patient_name, view=view)

            seq, seq_info = self.sitk_load_resize(patient_dir / seq_name, resize_size)
            seq_gt, seq_gt_info = self.sitk_load_resize(patient_dir / seq_gt_name, resize_size)

            assert seq_info['spacing'] == seq_gt_info['spacing']

            cfg = self.read_cfg(patient_dir / cfg_pattern.format(view=view))
            assert cfg['ED'] == cfg['NbFrame'] or cfg["ES"] == cfg['NbFrame']
            # let ed -> es
            if int(cfg['ED']) > int(cfg['ES']) :
                # print(patient_dir,view,':es -> ed')
                # flip video
                seq, seq_gt= np.flip(seq, axis=0), np.flip(seq_gt, axis=0)
                cfg['ED'], cfg['ES'] = cfg['ES'], cfg['ED']
                    
            # to rgb
            seq = np.repeat(seq[:, np.newaxis, :, :], 3, axis=1)

            frame_pairs_mask = {}
            for idxx, frame_gt in enumerate(seq_gt):
                frame_pairs_mask[str(idxx)] = frame_gt

            ef_cfg = cfg["EF"]
            # print(ef_cfg, ef)

            return seq, seq_gt, frame_pairs_mask, ef, edv, esv, seq_info['spacing']


    def compute_ef_to_patient(self, patient_dir:Path):
        patient_name = patient_dir.name 
        gt_mask_pattern = "{patient_name}_{view}_{instant}_gt.nii.gz"
        lv_label = 1

        view = "2CH"
        instant = "ED"
        a2c_ed, a2c_info = sitk_load(patient_dir / gt_mask_pattern.format(patient_name=patient_name, view=view, instant=instant))
        a2c_voxelspacing = a2c_info["spacing"][:2][::-1]    # Extract the (width,height) dimension from the metadata and order them like in the mask

        instant = "ES"
        a2c_es, _ = sitk_load(patient_dir / gt_mask_pattern.format(patient_name=patient_name, view=view, instant=instant))

        view = "4CH"
        instant = "ED"
        a4c_ed, a4c_info = sitk_load(patient_dir / gt_mask_pattern.format(patient_name=patient_name, view=view, instant=instant))
        a4c_voxelspacing = a4c_info["spacing"][:2][::-1]    # Extract the (width,height) dimension from the metadata and order them like in the mask

        instant = "ES"
        a4c_es, _ = sitk_load(patient_dir / gt_mask_pattern.format(patient_name=patient_name, view=view, instant=instant))
        # Extract binary LV masks from the multi-class segmentation masks
        a2c_ed_lv_mask = a2c_ed == lv_label
        a2c_es_lv_mask = a2c_es == lv_label
        a4c_ed_lv_mask = a4c_ed == lv_label
        a4c_es_lv_mask = a4c_es == lv_label

        # Use the provided implementation to compute the LV volumes
        edv, esv = compute_left_ventricle_volumes(a2c_ed_lv_mask, a2c_es_lv_mask, a2c_voxelspacing, a4c_ed_lv_mask, a4c_es_lv_mask, a4c_voxelspacing)
        ef = round(100 * (edv - esv) / edv, 2) # Round the computed value to the nearest integer

        # print(f"{patient_name=}: {ef=}, {edv=}, {esv=}")
        return ef, edv, esv
    
    def sitk_load_resize(self,filepath: Union[str, Path], resize_size) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Loads an image using SimpleITK and returns the image and its metadata.

        Args:
            filepath: Path to the image.

        Returns:
            - ([N], H, W), Image array.
            - Collection of metadata.
        """
        # Load image and save info
        image = sitk.ReadImage(str(filepath))
        image = self.resampleXYSize(image, *resize_size)
        info = {"origin": image.GetOrigin(), "spacing": image.GetSpacing(), "direction": image.GetDirection()}

        # Extract numpy array from the SimpleITK image object
        im_array = np.squeeze(sitk.GetArrayFromImage(image))

        return im_array, info

    def read_cfg(self, path):
        with open(path, 'r') as f:
            lines = f.readlines()

        data = {}

        for line in lines:
            line = line.strip()
            key, value = line.split(":")
            key = key.strip()
            value = value.strip()
            data[key] = value

        return data
    
    def resampleXYSize(self, sitkImage, new_xsize, new_ysize):
        '''
            newsitkImage = resampleSize(sitkImage, depth=DEPTH)
        '''
        #重采样函数
        euler3d = sitk.Euler3DTransform()

        xsize, ysize, zsize = sitkImage.GetSize()
        xspacing, yspacing, zspacing = sitkImage.GetSpacing()
        new_spacing_x = xspacing/(new_xsize/float(xsize))
        new_spacing_y = yspacing/(new_ysize/float(ysize))

        origin = sitkImage.GetOrigin()
        direction = sitkImage.GetDirection()
        #根据新的spacing 计算新的size
        newsize = (new_xsize,new_ysize,zsize)
        newspace = (new_spacing_x, new_spacing_y, zspacing)
        sitkImage = sitk.Resample(sitkImage,newsize,euler3d,sitk.sitkNearestNeighbor,origin,newspace,direction)
        return sitkImage

    def _get_video_filepath(self, patient_id:str):

        subdirectory_path = self.data_dir/"database_nifti"/patient_id
        video_filepath = subdirectory_path/(patient_id+self.suffix)
        annotation_filepath = subdirectory_path/self.annotation_filename
        # print(video_filepath)
        return video_filepath, annotation_filepath

        
    def generate_patients_id(self):
        patients_id = [f"patient0{i:03}" for i in range(1, 500+1)]
        train_size = 400
        val_size = 50
        test_size = 50

        # random.seed(1)
        # random.shuffle(patients_id)
        train_patients_id = patients_id[:train_size]
        val_patients_id = patients_id[train_size:train_size+val_size]
        test_patients_id = patients_id[-test_size:]

        return {
            "train": train_patients_id,
            "val": val_patients_id,
            "test": test_patients_id,
        }


    def get_data_split(self):

        split_dir = self.data_dir/"database_split"
        if self.mode == "train":
            split_file = split_dir/"subgroup_training.txt"
        elif self.mode == "val":
            split_file = split_dir/"subgroup_validation.txt"
        elif self.mode == "test":
            split_file = split_dir/"subgroup_testing.txt"
        
        with open(split_file, "r") as file:
            lines = file.readlines()
        lines = [line.strip() for line in lines]
        
        return lines

    def _read_cfg_file(self, path):
        config = configparser.ConfigParser()
        with open(path, 'r') as f:
            config_str = '[TOP]\n' + f.read()
        config_fp = StringIO(config_str)
        config.read_file(config_fp)
        return config

    


if __name__ == "__main__":

    data_dir = Path("/projects/p32335/my_research/echo_vision/data/camus/")
    # data_dir = Path("")

    dataset = CamusDataset(data_dir, mode="train", task="ef_regression")

    re = dataset[3]
    print(re[0].shape, re[1], re[2].shape)
