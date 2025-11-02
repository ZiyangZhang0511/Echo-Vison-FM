import os
import argparse
import time
import pickle
import random
from tqdm import tqdm
from pathlib import Path

from accelerate import Accelerator

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
# from pytorch_lightning import seed_everything

from datasets.mimic import MimicEchoDataset, MIMICECHOIterableDataset
from modeling.videomae import get_videomae_for_pretraining
from utilities.utils import remove_directory



PRETRAIN_CHECKPOINTS_DIR = "/projects/p32335/my_research/echo_vision/checkpoints/checkpoints_pretrain_new/"

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
             "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
             "and an Nvidia Ampere GPU.",
    )

    # parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--data_ratio", type=float, required=True)
    parser.add_argument("--mask_ratio", default=0.9, type=float)

    parser.add_argument("--initial_lr", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_epochs", default=50, type=int)
    parser.add_argument("--num_workers", default=8 , type=int)

    parser.add_argument("--restart_optimizer", action="store_true")
    parser.add_argument("--cur_epoch", default=-1, type=int)
    parser.add_argument("--resume_pretraining", action="store_true")

    args = parser.parse_args()

    return args


def train_function(model, dataloader, base_ckpt_dir, args):

    frames_count = 16
    num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
    seq_length = (frames_count // model.config.tubelet_size) * num_patches_per_frame

    dataset_size = len(dataloader.dataset.filespath)
    # print(dataset_size)

    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    optimizer = optim.AdamW(model.parameters(), lr=args.initial_lr, betas=(0.9, 0.98), weight_decay=0.05)
    # scheduler = CosineAnnealingLR(optimizer, (dataset_size//args.batch_size)*args.num_epochs, eta_min=1e-8)
    scheduler = LinearWarmupCosineAnnealingLR(
        optimizer,
        warmup_epochs=5*len(dataloader), 
        max_epochs=args.num_epochs*len(dataloader), 
        warmup_start_lr=1e-5, 
        eta_min=1e-8,
    )

    model, dataloader, optimizer, scheduler = accelerator.prepare(
         model, dataloader, optimizer, scheduler
    )
    # print(len(dataloader.dataset.filespath))

    if not args.resume_pretraining:
        cur_epoch = -1
        best_loss = 1e10
        accelerator.save_state(base_ckpt_dir + f"checkpoint_D{args.data_ratio}_M{args.mask_ratio}_E{cur_epoch}", safe_serialization=False)
    else: 
        cur_epoch = args.cur_epoch
        best_loss_filepath = base_ckpt_dir + f"best_loss_D{args.data_ratio}_M{args.mask_ratio}.pkl"
        with open(best_loss_filepath, 'rb') as file:
            best_loss = pickle.load(file)
        accelerator.print(f"current best loss: {best_loss}")
        accelerator.load_state(base_ckpt_dir + f"checkpoint_D{args.data_ratio}_M{args.mask_ratio}_E{cur_epoch}_best")

    if args.restart_optimizer:
        optimizer = optim.AdamW(model.parameters(), lr=args.initial_lr, betas=(0.9, 0.98), weight_decay=0.05)
        # scheduler = CosineAnnealingLR(optimizer, (dataset_size//args.batch_size)*args.num_epochs, eta_min=1e-8)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=5*len(dataloader), 
            max_epochs=args.num_epochs*len(dataloader), 
            warmup_start_lr=1e-5, 
            eta_min=1e-8,
        )
        optimizer, scheduler = accelerator.prepare(optimizer, scheduler)

    accelerator.print("Starting the training model...")
    # step = 0
    for epoch in range(cur_epoch+1, args.num_epochs):
        model.train()
        epoch_loss = 0
        for i, batch in enumerate(tqdm(dataloader)):
            pixel_values, _ = batch
            pixel_values = pixel_values.to(accelerator.device)
            cur_batch_size = pixel_values.shape[0]
            # print(cur_batch_size)

            # bool_masked_pos = torch.randint(0, 2, (1, seq_length)).repeat(cur_batch_size, 1).bool()

            random_tensor = torch.rand((1, seq_length))
            bool_masked_pos = (random_tensor > args.mask_ratio)
            bool_masked_pos = bool_masked_pos.repeat(cur_batch_size, 1).bool()
            # print(bool_masked_pos.sum(dim=1) / seq_length)
            # return

            outputs = model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)
            loss = outputs.loss
            
            del outputs.logits
            # gc.collect()
            
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item() / len(dataloader)
            # if step % 1000 == 0 and epoch < 5:
            #     accelerator.print(f"step {step}, current lr: {scheduler.get_lr()}, running loss: {loss.item()}")
            
            # step += 1    
            
        accelerator.print(f"Epoch {epoch}, current_lr: {scheduler.get_last_lr()}, train Loss: {epoch_loss}")
        accelerator.wait_for_everyone()

        if best_loss > epoch_loss:
            best_loss = epoch_loss
            with open(base_ckpt_dir + f"best_loss_D{args.data_ratio}_M{args.mask_ratio}.pkl", 'wb') as file:
                pickle.dump(best_loss, file)
        
            accelerator.save_state(base_ckpt_dir + f"checkpoint_D{args.data_ratio}_M{args.mask_ratio}_E{epoch}_best", safe_serialization=False)
            accelerator.print(f"saved a better ckpt at {epoch}")

        elif accelerator.is_main_process: 
            accelerator.save_state(base_ckpt_dir + f"checkpoint_D{args.data_ratio}_M{args.mask_ratio}_E{epoch}", safe_serialization=False)
            accelerator.print(f"saved at epoch {epoch}")
        accelerator.wait_for_everyone()

        if epoch > 0 and accelerator.is_main_process:
            dir_path = base_ckpt_dir + f"checkpoint_D{args.data_ratio}_M{args.mask_ratio}_E{epoch-2}"
            if os.path.exists(dir_path):
                remove_directory(dir_path)
        accelerator.wait_for_everyone()
        

def main():
    args = get_args()

    ###========get pretrained dataset and dataloader========###
    base_ckpt_dir = PRETRAIN_CHECKPOINTS_DIR
    os.makedirs(base_ckpt_dir, exist_ok=True)

    data_dir = "/home/olg7848/p32335/MIMIC-ECHO/mimic_echo_avi"
    data_dir = Path(data_dir)

    dataset = MimicEchoDataset(
        data_dir,
        frames_count=16,
        data_ratio=args.data_ratio,
        decord=True,
    )

    print(len(dataset))
    
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        # drop_last=False,
        num_workers=args.num_workers,
        # prefetch_factor=1,
        # pin_memory=False,
        # persistent_workers=False,
    )

    ###========get model========###
    model = get_videomae_for_pretraining()

    ###========train model on mimic dataset and save checkpoint========###
    train_function(model, dataloader, base_ckpt_dir, args)



if __name__ == "__main__":
    main()