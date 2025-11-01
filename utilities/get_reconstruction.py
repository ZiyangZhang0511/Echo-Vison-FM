from pathlib import Path
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image

from transformers import (
        VideoMAEConfig, 
        VideoMAEForPreTraining,
)

from accelerate import Accelerator

from datasets.mimic import MimicEchoDataset


if __name__ == "__main__":

    ###========get videomae for pretraining========###
    configuration = VideoMAEConfig()
    configuration.image_size = 224
    model = VideoMAEForPreTraining.from_pretrained(
        "MCG-NJU/videomae-base",
        config=configuration,
    )

    pretrained_checkpiont_path = "/home/olg7848/p32335/my_research/echo_vision/checkpoints/checkpoints_pretrain/half/checkpoint_50"
    accelerator = Accelerator()
    model = accelerator.prepare(model)
    accelerator.load_state(pretrained_checkpiont_path)

    frames_count = 16
    num_patches_per_frame = (model.config.image_size // model.config.patch_size) ** 2
    seq_length = (frames_count // model.config.tubelet_size) * num_patches_per_frame

    ###========get sub dataset========###
    data_dir = "/home/olg7848/olg7848/MIMIC_ECHO/p10"
    data_dir = Path(data_dir)
    mimic_dataset = MimicEchoDataset(data_dir, frames_count=16)
    indices = list(range(0, 1))
    subset = Subset(mimic_dataset, indices)
    dataloader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    model.eval()
    for i, batch in enumerate(tqdm(dataloader)):
        pixel_values, _ = batch
        pixel_values = pixel_values.to(accelerator.device)
        print(pixel_values.size())
        cur_batch_size = pixel_values.shape[0]

        


        
        bool_masked_pos = torch.randint(0, 2, (1, seq_length)).repeat(cur_batch_size, 1).bool()
        print(bool_masked_pos.sum(), seq_length)
        print(bool_masked_pos)

        outputs = model(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos, return_dict=True)
        logits = outputs.logits
        print(logits.size())

        pixel_values_sq = pixel_values.view(cur_batch_size, 8, 2, 3, 14, 16, 14, 16)

        # Step 2: Permute dimensions to bring the patch-related dimensions together
        # [1, 16, 3, 14, 14, 16, 16]
        pixel_values_sq = pixel_values_sq.permute(0, 4, 6, 1, 5, 7, 3, 2).contiguous()

        # Step 3: Reshape to combine dimensions into the desired shape
        # [1, 1568 (14*14*8), 1536 (16*16*3*2)]
        pixel_values_sq = pixel_values_sq.reshape(cur_batch_size, 14*14*8, 16*16*3*2)

        print(pixel_values_sq.size())

        for b in range(cur_batch_size):
            # Get the indices where the mask is True
            indices = torch.nonzero(bool_masked_pos[b], as_tuple=True)[0]
            print(indices.size())
            pixel_values_sq[b, indices] = logits

        print(pixel_values_sq.size())

        pixel_values_recon = pixel_values_sq.view(cur_batch_size, 14, 14, 8, 16, 16, 3, 2)

        # Step 2: Permute to place the patch-related dimensions back together
        # From [1, 14, 14, 8, 16, 16, 3, 2] to [1, 16, 3, 14, 16, 14, 16]
        pixel_values_recon = pixel_values_recon.permute(0, 3, 7, 6, 1, 4, 2, 5).contiguous()

        # Step 3: Reshape to the desired shape
        # From [1, 16, 3, 14, 16, 14, 16] to [1, 16, 3, 224, 224]
        pixel_values_recon = pixel_values_recon.reshape(cur_batch_size, 16, 3, 224, 224)

        print(pixel_values_recon.size())


        save_image(pixel_values_recon[0].detach().cpu(), "./pixel_values_recon.jpg", nrow=4)
        save_image(pixel_values[0].detach().cpu(), "./pixel_values_real.jpg", nrow=4)

        