import os
import argparse
from pathlib import Path
from tqdm.auto import tqdm

import torch
import torch.nn as nn
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
        choices=["vivit", "videoresnet", "vanilla_videomae", "echo_videomae", "raw_videomae"],
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
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    return args


def main():

    args = get_args()

    ###========get test dataset========###
    # test_dataset = EchonetDynamicDataset(DATAPATH_DICT[args.dataset_name], mode="test", task=args.task_type)
    # test_dataloader = DataLoader(
    #     test_dataset,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     drop_last=False,
    #     num_workers=args.num_workers
    # )

    if args.dataset_name == "echonet_dynamic":
        val_dataset = EchonetDynamicDataset(DATAPATH_DICT[args.dataset_name], mode="val", task=args.task_type)
        test_dataset = EchonetDynamicDataset(DATAPATH_DICT[args.dataset_name], mode="test", task=args.task_type)
    elif args.dataset_name == "camus":
        val_dataset = CamusDataset(DATAPATH_DICT[args.dataset_name], mode="val", task=args.task_type)
        test_dataset = CamusDataset(DATAPATH_DICT[args.dataset_name], mode="test", task=args.task_type)
    elif args.dataset_name == "tmed":
        val_dataset = TMED2Dataset(DATAPATH_DICT[args.dataset_name], mode="val", task=args.task_type)
        test_dataset = TMED2Dataset(DATAPATH_DICT[args.dataset_name], mode="test", task=args.task_type)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
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
        print(checkpoint["epoch"], checkpoint["val_loss"])

    ###======== test model for performance ========###
    print("Starting test model...")
    model.eval()

    metrics_dict = utils.run_multiple_tests(
        model,
        test_dataloader,
        criterion,
        "test",
        args,
        args.device,
        save_for_boxplot=False,
    )
    print(metrics_dict)

    test_loss, test_metrics_dict = utils.test(
        None,
        model,
        test_dataloader,
        criterion,
        "test",
        args,
        is_save_results=False,
    )
    print(test_metrics_dict)


if __name__ == "__main__":

    main()