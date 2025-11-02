import argparse
import os
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm
import time

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from accelerate import Accelerator

from utilities import utils

from datasets.echonet import EchonetDynamicDataset
from datasets.camus import CamusDataset

from modeling.classifier import VideoBinaryClassifier
from modeling.regressor import VideoRegressor

DATAPATH_DICT = {"echonet_dynamic": Path("/projects/p32335/Research_on_AI_medicine/EchoGPT/VideoEncoder/Data/EchoNet-Dynamic"),
                 "camus": Path("/projects/p32335/Research_on_AI_medicine/EchoGPT/VideoEncoder/Data/CAMUS"),
}

CHECKPOINTS_DIR = "./checkpoints/checkpoints_baseline/"

def parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_name", 
        default=None, 
        type=str, 
        choices=["echonet_dynamic", "camus"],                
        help="downstream task dataset to be finetuned.",
    )
    parser.add_argument(
        "--feat_extractor", 
        default=None, 
        type=str, 
        choices=["vivit", "videoresnet", "vanilla_videomae"],
    )
    parser.add_argument(
        "--task_type", 
        default=None, 
        type=str, 
        choices=["ef_regression", "esv_regression", "edv_regression", "ef_classification"],
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
             "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
             "and an Nvidia Ampere GPU.",
    )

    parser.add_argument("--initial_lr", default=1e-5, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_epochs", default=50, type=int)
    parser.add_argument("--num_workers", default=8, type=int)

    args = parser.parse_args()

    return args

def train_function(model, train_dataloader, val_dataloader, test_dataloader, args):

    if args.task_type in ["ef_regression", "esv_regression", "edv_regression"]:
        criterion = nn.MSELoss()
    elif args.task_type == "ef_classification":
        criterion = nn.BCEWithLogitsLoss()
        
    optimizer = optim.AdamW(model.parameters(), lr=args.initial_lr)
    # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=10, 
        max_epochs=args.num_epochs, 
        warmup_start_lr=1e-5, 
        eta_min=1e-8,
    )

    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    model, train_dataloader, val_dataloader, test_dataloader, optimizer, scheduler = accelerator.prepare(
        model, train_dataloader, val_dataloader, test_dataloader, optimizer,  scheduler
    )

    best_val_loss = 100000

    accelerator.print("Starting the training model...")
    for epoch in range(args.num_epochs):

        ###========training process========###
        model.train()
        epoch_logits, epoch_targets = [], []
        epoch_loss = 0
        
        for i, (pixel_values, targets, temporal_indices) in enumerate(tqdm(train_dataloader)):
            pixel_values = pixel_values.to(accelerator.device)
            temporal_indices = temporal_indices.to(accelerator.device)
            targets = targets.view(-1, 1).to(accelerator.device)
            logits = model(pixel_values, temporal_indices)
            
            loss = criterion(logits, targets)
            accelerator.backward(loss)
            optimizer.step()
            # scheduler.step(i + epoch / len(train_dataloader))
            optimizer.zero_grad()

            epoch_loss += loss.item() / len(train_dataloader)
            epoch_logits.append(accelerator.gather(logits))
            epoch_targets.append(accelerator.gather(targets))

        scheduler.step()

        epoch_logits = torch.cat(epoch_logits).detach().cpu()
        epoch_targets = torch.cat(epoch_targets).detach().cpu()

        train_metrics_dict = utils.compute_metrics(epoch_logits, epoch_targets, args.task_type)
        accelerator.print(f"Epoch {epoch}, current lr: {scheduler.get_last_lr()}, train loss: {epoch_loss}, train metrics: {train_metrics_dict}")

        ###========validation process========###
        val_loss, _ = utils.test(accelerator, model, val_dataloader, criterion, "val", args)

        ###========save checkpoint========###
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(CHECKPOINTS_DIR, f"best_epoch_{epoch}_{args.feat_extractor}_{args.dataset_name}_{args.task_type}")
            accelerator.save_state(checkpoint_path)
            accelerator.print(f"saved checkpoint at epoch {epoch}.")
        
    time.sleep(30)

    ###========test the best checkpoint and output result to a .txt file========###
    bestcp_path = utils.get_the_most_recent_dir(CHECKPOINTS_DIR)
    accelerator.load_state(bestcp_path)

    accelerator.print("Starting test model...")
    test_loss, test_metrics_dict = utils.test(accelerator, model, test_dataloader, criterion, "test", args)

    info = f"test loss: {test_loss}, test metrics: {test_metrics_dict} on {os.path.basename(bestcp_path)}"
    result_filepath = f"./results/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    with open(result_filepath, 'w') as f:
        f.write(info)


def main():
    print("Executing this script...")
    args = parser()

    ###========get dataset and dataloader to be finetuned=======###
    if args.dataset_name == "echonet_dynamic":
        train_dataset = EchonetDynamicDataset(DATAPATH_DICT[args.dataset_name], mode="train", task=args.task_type)
        val_dataset = EchonetDynamicDataset(DATAPATH_DICT[args.dataset_name], mode="val", task=args.task_type)
        test_dataset = EchonetDynamicDataset(DATAPATH_DICT[args.dataset_name], mode="test", task=args.task_type)
    elif args.dataset_name == "camus":
        train_dataset = CamusDataset(DATAPATH_DICT[args.dataset_name], mode="train", task=args.task_type)
        val_dataset = CamusDataset(DATAPATH_DICT[args.dataset_name], mode="val", task=args.task_type)
        test_dataset = CamusDataset(DATAPATH_DICT[args.dataset_name], mode="test", task=args.task_type)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, drop_last=False, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=False, num_workers=args.num_workers)

    ###========get model to be finetuned========###
    if args.task_type == "ef_classification":
        model = VideoBinaryClassifier(args.feat_extractor)
    elif args.task_type in ["ef_regression", "esv_regression", "edv_regression"]:
        model = VideoRegressor(args.feat_extractor)
        
    ###========train model and save chechpoint and test the best model========###
    train_function(model, train_dataloader, val_dataloader, test_dataloader, args)


if __name__ == "__main__":
    main()
