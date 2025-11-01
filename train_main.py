import os
import time
import argparse
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, ConcatDataset, Subset

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR

from accelerate import Accelerator

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

CHECKPOINTS_DIR = "./checkpoints/checkpoints_main/"


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
    parser.add_argument("--data_ratio", type=float, default=1.0)

    parser.add_argument("--initial_lr", default=1e-4, type=float)
    parser.add_argument("--warmup_start_lr", default=1e-5, type=float)
    parser.add_argument("--warmup_epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--num_epochs", default=50, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--resume_train", action="store_true")
    parser.add_argument("--batch_step_lr", action="store_false")

    parser.add_argument("--pretrained_ckpt_path", default=None, type=str)
    parser.add_argument("--ckpt_save_dir", required=True, type=str)

    args = parser.parse_args()

    return args

def train_function(model, train_dataloader, val_dataloader, test_dataloader, args):

    if args.task_type in ["ef_regression", "esv_regression", "edv_regression"]:
        criterion = nn.MSELoss()
    elif args.task_type == "ef_classification":
        criterion = nn.BCEWithLogitsLoss()
    elif args.task_type == "as_classification":
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.AdamW(model.parameters(), lr=args.initial_lr, betas=(0.9, 0.99), weight_decay=1e-5)
    # scheduler = CosineAnnealingLR(optimizer, len(train_dataloader)*args.num_epochs, eta_min=1e-8)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=args.warmup_epochs*len(train_dataloader) if args.batch_step_lr else args.warmup_epochs, 
        max_epochs=args.num_epochs*len(train_dataloader) if args.batch_step_lr else args.num_epochs, 
        warmup_start_lr=args.warmup_start_lr,
        eta_min=1e-8,
    )

    # accelerator = Accelerator(mixed_precision=args.mixed_precision)
    # model, train_dataloader, val_dataloader, test_dataloader, optimizer, scheduler = accelerator.prepare(
    #     model, train_dataloader, val_dataloader, test_dataloader, optimizer,  scheduler
    # )

    best_val_loss = 1e10
    model.to(args.device)

    checkpoint_path = os.path.join(
        args.ckpt_save_dir,
        f"{args.feat_extractor}_{args.dataset_name}_{args.task_type}_stff{args.stff_net_flag}_linearhead{args.linear_prediction_head}_finetune{args.finetune}.pth"
    )

    if args.resume_train:
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        scheduler = CosineAnnealingLR(optimizer, len(train_dataloader)*args.num_epochs, eta_min=1e-8)
        # scheduler = LinearWarmupCosineAnnealingLR(
        #     optimizer,
        #     warmup_epochs=args.warmup_epochs*len(train_dataloader) if args.batch_step_lr else args.warmup_epochs, 
        #     max_epochs=args.num_epochs*len(train_dataloader) if args.batch_step_lr else args.num_epochs, 
        #     warmup_start_lr=args.warmup_start_lr,
        #     eta_min=1e-8,
        # )

    print("Starting the training model...")
    for epoch in range(args.num_epochs):
        # break
        ###======== training process ========###
        model.train()
        epoch_logits, epoch_targets = [], []
        epoch_loss = 0
        
        for i, (pixel_values, targets, temporal_indices) in enumerate(tqdm(train_dataloader)):
            pixel_values = pixel_values.to(args.device)

            if args.task_type == "as_classification":
                targets = targets.to(args.device)
            else:
                targets = targets.view(-1, 1).to(args.device)

            logits = model(pixel_values, temporal_indices)
            
            loss = criterion(logits, targets)
            
            optimizer.zero_grad()
            loss.backward()
            # accelerator.backward(loss)
            optimizer.step()
            
            if args.batch_step_lr:
                scheduler.step()
            else:
                scheduler.step(i + epoch / len(train_dataloader))

            epoch_loss += loss.item() / len(train_dataloader)

            # epoch_logits.append(accelerator.gather(logits))
            # epoch_targets.append(accelerator.gather(targets))

            epoch_logits.append(logits)
            epoch_targets.append(targets)
            
            # break

        epoch_logits = torch.cat(epoch_logits).detach().cpu()
        epoch_targets = torch.cat(epoch_targets).detach().cpu()

        train_metrics_dict = utils.compute_metrics(epoch_logits, epoch_targets, args.task_type)
        print(f"Epoch {epoch}, current lr: {scheduler.get_last_lr()}, train loss: {epoch_loss}, train metrics: {train_metrics_dict}")

        ###======== validation process ========###
        model.eval()
        val_loss, _ = utils.test(None, model, val_dataloader, criterion, "val", args)

        ###======== save checkpoint ========###
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            # accelerator.save_state(checkpoint_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                # 'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"saved checkpoint at epoch {epoch}.")

    ###========test the best checkpoint and output result to a .txt file========###
    # best_ckpt_path = utils.get_the_most_recent_dir(args.ckpt_save_dir)
    # accelerator.load_state(best_ckpt_path)

    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Starting test model...")
    model.eval()
    # test_loss, test_metrics_dict = utils.test(None, model, test_dataloader, criterion, "test", args)

    metrics_dict = utils.run_multiple_tests(
        model,
        test_dataloader,
        criterion,
        "test",
        args,
        args.device,
    )
    print(metrics_dict)

    # info = f"test loss: {test_loss}, test metrics: {test_metrics_dict} on {os.path.basename(bestcp_path)}"
    # result_filepath = f"./results/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"
    # with open(result_filepath, 'w') as f:
    #     f.write(info)


def main():
    print("Executing this script......")
    args = get_args()
    os.makedirs(args.ckpt_save_dir, exist_ok=True)

    ###========get dataset and dataloader to be finetuned=======###
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
    
    subset = Subset(test_dataset, indices=range(0, len(test_dataset)//2))
    # train_dataset = ConcatDataset([train_dataset, test_dataset])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, drop_last=False, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, drop_last=False, num_workers=args.num_workers)

    print(len(train_dataset), len(val_dataset), len(test_dataset))

    ###========get model to be finetuned========###
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

    ###========train model and save chechpoint and test the best model========###
    train_function(model, train_dataloader, val_dataloader, test_dataloader, args)


if __name__ == "__main__":

    main()
