import os
import time
import argparse
from pathlib import Path

import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from torchprofile import profile_macs
from torchprofile.utils.trace import trace as _raw_trace
# from torchprofile.compute_macs import compute_macs

from fvcore.nn import FlopCountAnalysis, flop_count_table, parameter_count_table

from utilities import utils

from datasets.echonet import EchonetDynamicDataset
from datasets.camus import CamusDataset
from datasets.tmed import TMED2Dataset

from modeling.classifier import VideoBinaryClassifier
from modeling.regressor import VideoRegressor

DATAPATH_DICT = {
    "echonet_dynamic": Path("/home/olg7848/p32335/my_research/echo_vision/data/echonet_dynamic"),
    "camus": Path("/home/olg7848/p32335/my_research/echo_vision/data/camus"),
    "tmed": Path("/home/olg7848/p32335/my_research/echo_vision/data/TMED2/approved_users_only"),
}

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_name", default="echonet_dynamic", type=str, choices=["echonet_dynamic", "camus", "tmed"], help="downstream task dataset to be finetuned.")
    parser.add_argument("--feat_extractor", default="echo_videomae", type=str, choices=["vivit", "videoresnet", "vanilla_videomae", "echo_videomae"])
    parser.add_argument("--task_type", default="ef_regression", type=str, choices=["ef_regression", "esv_regression", "edv_regression", "ef_classification", "as_classification"],)
    parser.add_argument("--pretrained_ckpt_path", default=None, type=str)
    parser.add_argument("--stff_net_flag", action="store_true")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--linear_prediction_head", action="store_true")
    parser.add_argument("--data_ratio", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    return args



def main():
    args = get_args()

    ### build model ###
    if args.task_type == "ef_classification":
        model = VideoBinaryClassifier(
            args.feat_extractor,
            args.pretrained_ckpt_path,
            args.stff_net_flag,
            args.linear_prediction_head,
            args.finetune,
        )

    elif args.task_type in ["ef_regression", "esv_regression", "edv_regression"]:
        model = VideoRegressor(
            args.feat_extractor,
            args.pretrained_ckpt_path,
            args.stff_net_flag,
            args.linear_prediction_head,
            args.finetune,
        )
    
    elif args.task_type in ["as_classification"]:
        model = VideoBinaryClassifier(
            args.feat_extractor,
            args.pretrained_ckpt_path,
            args.stff_net_flag,
            args.linear_prediction_head,
            args.finetune,
            num_classes=5,
        )
    model.to(args.device)
    model.eval()
    # print(model)

    ### get dataloader ###
    if args.dataset_name == "echonet_dynamic":
        train_dataset = EchonetDynamicDataset(DATAPATH_DICT[args.dataset_name], mode="train", task=args.task_type, data_ratio=args.data_ratio)
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
    
    # subset = Subset(test_dataset, indices=range(0, len(test_dataset)//2))
    # train_dataset = ConcatDataset([train_dataset, test_dataset])

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, drop_last=False, num_workers=8)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, drop_last=False, num_workers=8)
    # print(len(train_dataset), len(val_dataset), len(test_dataset))

    ### calculate MACs FLOPs ###

    first_batch = next(iter(train_dataloader))
    # print(first_batch[0].size())
    # for key in first_batch.keys():
    #     first_batch[key] = first_batch[key].to(args.device)
    batch_size = first_batch[0].shape[0]
    flops = FlopCountAnalysis(model, (first_batch[0].to(args.device), first_batch[2].to(args.device))).total() / batch_size / 1e9
    macs = flops / 2 
    print(f"MACs: {macs:.2f}G, FLOPs: {flops:.2f}G")

    ### count trainable parameters ###
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000
    print(f"Number of trainable parameters: {num_trainable_params:.2f}M")
    if args.stff_net_flag:
        num_trainable_params = sum(p.numel() for p in model.stff_net.parameters() if p.requires_grad) / 1000000
        print(f"Number of trainable parameters on stf_net: {num_trainable_params:.2f}M")
        print("\n=== Parameter table ===")
        print(parameter_count_table(model))

    ### compute Throughput ###
    warmup, iters = 10, 100   
    #warm‑up
    for _ in range(warmup):
        step(model, (first_batch[0].to(args.device), first_batch[2].to(args.device)))

    #timed loop
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        step(model, (first_batch[0].to(args.device), first_batch[2].to(args.device)))
    torch.cuda.synchronize()

    elapsed = time.time() - t0
    throughput = iters * batch_size / elapsed
    print(f"Throughput: {throughput:.2f} samples/s")

    speed = 1000 * (1/throughput)
    print(f"Inference speed: {speed:.2f} ms/samples")

def step(model, batch):
    with autocast():           
        out = model(*batch)

if __name__ == "__main__":
    main()