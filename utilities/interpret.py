import os
import argparse
from pathlib import Path
from tqdm.auto import tqdm

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from accelerate import Accelerator

from datasets.echonet import EchonetDynamicDataset
from datasets.camus import CamusDataset
from datasets.tmed import TMED2Dataset

from modeling.classifier import VideoBinaryClassifier
from modeling.regressor import VideoRegressor

from utilities import utils

DATAPATH_DICT = {
    # "echonet_dynamic": Path("/projects/p32335/Research_on_AI_medicine/EchoGPT/VideoEncoder/Data/EchoNet-Dynamic"),
    # "camus": Path("/projects/p32335/Research_on_AI_medicine/EchoGPT/VideoEncoder/Data/CAMUS"),
    "echonet_dynamic": Path("/home/olg7848/p32335/my_research/echo_vision/data/echonet_dynamic"),
    "camus": Path("/home/olg7848/p32335/my_research/echo_vision/data/camus"),
    "tmed": Path("/home/olg7848/p32335/my_research/echo_vision/data/TMED2/approved_users_only"),
}

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_name", 
        default="echonet_dynamic", 
        type=str, 
        choices=["echonet_dynamic", "camus", "tmed"],                
        help="downstream task dataset to be finetuned.",
    )
    parser.add_argument(
        "--feat_extractor", 
        default="echo_videomae", 
        type=str, 
        choices=["vivit", "videoresnet", "vanilla_videomae", "echo_videomae"],
    )
    parser.add_argument(
        "--task_type", 
        default="ef_regression", 
        type=str, 
        choices=["ef_regression", "esv_regression", "edv_regression", "ef_classification", "as_classification"],
    )

    parser.add_argument(
        "--ckpt_path", 
        type=str,
    )

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
             "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
             "and an Nvidia Ampere GPU.",
    )

    parser.add_argument("--stff_net_flag", action="store_true")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--linear_prediction_head", action="store_true")

    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    return args

def rollout(
    attentions,
    discard_ratio=0.9,
    head_fusion="min",
    mode="3D",
):
    result = torch.eye(attentions[0].size(-1)).to(attentions[0].device)
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"
            # print("attention_heads_fused:", attention_heads_fused.shape)
            # return

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            # print("flat:", flat.shape)
            # return

            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1)).to(attention_heads_fused.device)
            a = (attention_heads_fused + 1.0*I)/2
            # print(a.sum(dim=-1).shape)
            # return
            a = a / a.sum(dim=-1, keepdim=True)


            result = torch.matmul(a, result)

    # Look at the total attention between the class token,
    # and the image patches

    # In case of 224x224 image, this brings us from 196 to 14
    if mode == "2D":
        mask = result[0, 0, 1:]
        width = int(mask.size(-1)**0.5)
        mask = mask.reshape(width, width).detach().cpu()
    elif mode == "3D":
        seq_len = attentions[0].size(-1)
        mask = result[0, 125, :]
        # mask = result[0].mean(1)
        mask = mask.reshape(8, 14, 14).detach().cpu()
    mask = (mask - mask.min()) / (mask.max() - mask.min())

    return mask

def save_heatmap(video, mask, save_stem):
    os.makedirs("./check_data/heatmap/"+save_stem, exist_ok=True)
    mask_up = F.interpolate(
        mask.unsqueeze(1), size=(224, 224),
        mode='bilinear', align_corners=False,
    ).squeeze(1)  # (8,224,224)

    for i in range(video.shape[0]):          # 这里默认掩码通道 i 对应视频第 i 帧
        frame_idx = i                        # 如需其他对应关系，请自行修改
        frame = video[frame_idx].cpu().numpy() # (3, 224,224)
        heat  = mask_up[i//2].cpu().numpy()       # (224,224)
        # heat = mask_up.mean(0).cpu().numpy()
        fig, ax = plt.subplots(figsize=(4,4))
        ax.imshow(frame.transpose(1, 2, 0))          # 原始帧灰度
        ax.imshow(heat,  cmap='jet', alpha=0.5)# 掩码热图半透明叠加
        ax.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig("./check_data/heatmap/"+save_stem+f"/overlay_{i}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)


def main():

    args = get_args()

    ###======== get dataloader ========###
    if args.dataset_name == "echonet_dynamic":
        train_dataset = EchonetDynamicDataset(DATAPATH_DICT[args.dataset_name], mode="train", task=args.task_type)
        val_dataset = EchonetDynamicDataset(DATAPATH_DICT[args.dataset_name], mode="val", task=args.task_type)
        test_dataset = EchonetDynamicDataset(DATAPATH_DICT[args.dataset_name], mode="test", task=args.task_type)
    elif args.dataset_name == "camus":
        train_dataset = CamusDataset(DATAPATH_DICT[args.dataset_name], mode="train", task=args.task_type)
        val_dataset = CamusDataset(DATAPATH_DICT[args.dataset_name], mode="val", task=args.task_type)
        test_dataset = CamusDataset(DATAPATH_DICT[args.dataset_name], mode="test", task=args.task_type)
    elif args.dataset_name == "tmed":
        train_dataset = TMED2Dataset(DATAPATH_DICT[args.dataset_name], mode="train", task=args.task_type)
        val_dataset = TMED2Dataset(DATAPATH_DICT[args.dataset_name], mode="val", task=args.task_type)
        test_dataset = TMED2Dataset(DATAPATH_DICT[args.dataset_name], mode="test", task=args.task_type)


    g = torch.Generator()
    g.manual_seed(3)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        generator=g,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        generator=g,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        generator=g,
    )

    ###======== get model with weights ========###
    if args.task_type == "ef_classification":
        model = VideoBinaryClassifier(
            args.feat_extractor,
            stff_net_flag=args.stff_net_flag,
            linear_prediction_head=args.linear_prediction_head,
        )
        criterion = nn.BCEWithLogitsLoss()
        
    elif args.task_type in ["ef_regression", "esv_regression", "edv_regression"]:
        model = VideoRegressor(
            args.feat_extractor,
            stff_net_flag=args.stff_net_flag,
            linear_prediction_head=args.linear_prediction_head,
        )
        criterion = nn.MSELoss()
    
    elif args.task_type in ["as_classification"]:
        model = VideoBinaryClassifier(
            args.feat_extractor,
            stff_net_flag=args.stff_net_flag,
            linear_prediction_head=args.linear_prediction_head,
            num_classes=5,
        )
        criterion = nn.CrossEntropyLoss()
    
    model.to(args.device)
        

    if os.path.isdir(args.ckpt_path):
        accelerator = Accelerator()
        model = accelerator.prepare(model)
        accelerator.load_state(args.ckpt_path)
        model = accelerator.unwrap_model(model)

    elif os.path.isfile(args.ckpt_path):
        checkpoint = torch.load(args.ckpt_path, weights_only=False, map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    
    first_batch = next(iter(val_dataloader))
    outputs = model.pretrained_encoder(
        pixel_values=first_batch[0].to(args.device),
        output_attentions=True,
    )
    self_attentions = outputs.attentions
    # print(len(self_attentions), self_attentions[-1].shape)

    mask = rollout(self_attentions)
    # print(mask.shape)



    video = first_batch[0][0].cpu().clone()          # (16,224,224)
    # print(video.size())
    # print(mask.size())
    save_stem = f"{args.feat_extractor}_{args.dataset_name}_{args.task_type}_stff{args.stff_net_flag}_linearhead{args.linear_prediction_head}_finetune{args.finetune}"
    save_heatmap(video, mask, save_stem)



def get_attention(attentions):
    attention = attentions[-1][0]
    return


if __name__ == "__main__":
    main()