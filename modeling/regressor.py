from transformers import (
        VideoMAEModel,
        VideoMAEConfig,
        VivitModel, 
        VivitConfig,
)

import torch
from torch import nn
from torchvision.models.video import r3d_18, R3D_18_Weights

from .videomae import get_echo_encoder
from .stff_net import SpatioTemporalFeatureFusionNet

class VideoRegressor(nn.Module):
    def __init__(
        self,
        feat_extractor:str,
        pretrained_checkpiont_path=None,
        stff_net_flag:bool=False,
        linear_prediction_head:bool=True,
        finetune:bool=True,
    ):
        super(VideoRegressor, self).__init__()
        self._feat_extractor = feat_extractor
        self._stff_net_flag = stff_net_flag

        if self._feat_extractor == "vanilla_videomae":
            self.pretrained_encoder = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
            config = self.pretrained_encoder.config
            hidden_size = config.hidden_size

        elif self._feat_extractor == "vivit":
            config = VivitConfig(num_frames=16)
            self.pretrained_encoder = VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400", config=config, ignore_mismatched_sizes=True)
            # print(self.pretrained_encoder)

            nn.init.xavier_uniform_(self.pretrained_encoder.pooler.dense.weight)
            nn.init.zeros_(self.pretrained_encoder.pooler.dense.bias)
            
            hidden_size = self.pretrained_encoder.config.hidden_size

        elif self._feat_extractor == "videoresnet":
            self.pretrained_encoder = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
            hidden_size = 400

        elif self._feat_extractor == "echo_videomae":
            self.pretrained_encoder = get_echo_encoder(pretrained_checkpiont_path)
            hidden_size = self.pretrained_encoder.config.hidden_size
        
        elif self._feat_extractor == "raw_videomae":
            config = VideoMAEConfig()
            self.pretrained_encoder = VideoMAEModel(config)
            hidden_size = self.pretrained_encoder.config.hidden_size

        else:
            ValueError("the value of 'feat_extractor' is wrong!!!")
        

        if self._stff_net_flag:
            self.stff_net = SpatioTemporalFeatureFusionNet(feat_dim=768, size_feat_map=(8, 14, 14))
            hidden_size = 768 * 2


        if linear_prediction_head:
            self.regressor = self.make_regressor_block(hidden_size, 1, final_layer=True)
        else:
            self.regressor = nn.Sequential(
                self.make_regressor_block(hidden_size, 200),
                self.make_regressor_block(200, 200 // 2),
                self.make_regressor_block(200 // 2, 1, final_layer=True),
            )


        if not finetune:
            self.freeze_feature_extractor()

    
    def forward(self, pixel_values, temporal_indices):
        
        if self._feat_extractor == "vanilla_videomae":
            video_embeddings = self.pretrained_encoder(pixel_values).last_hidden_state
            # video_embeddings = torch.max(video_embeddings, dim=1)[0]
            if self._stff_net_flag:
                # print(video_embeddings.size(), temporal_indices.size())
                video_embeddings = self.stff_net(video_embeddings, temporal_indices)
            else:
                video_embeddings = torch.max(video_embeddings, dim=1)[0]
        elif self._feat_extractor == "vivit":
            video_embeddings = self.pretrained_encoder(pixel_values).pooler_output
        elif self._feat_extractor == "videoresnet":
            video_embeddings = self.pretrained_encoder(pixel_values.permute(0, 2, 1, 3, 4).contiguous())
        elif self._feat_extractor in ["raw_videomae", "echo_videomae"]:
            video_embeddings = self.pretrained_encoder(pixel_values).last_hidden_state

            if self._stff_net_flag:
                # print(video_embeddings.size(), temporal_indices.size())
                video_embeddings = self.stff_net(video_embeddings, temporal_indices)
            else:
                video_embeddings = torch.max(video_embeddings, dim=1)[0]
        
        logits = self.regressor(video_embeddings)
        return logits

    def make_regressor_block(self, input_channels, output_channels, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Linear(input_channels, output_channels),
                nn.BatchNorm1d(output_channels),
                # nn.LayerNorm(output_channels),
                nn.ReLU()
            )
        else:
            return nn.Sequential(
                nn.Linear(input_channels, output_channels)
            )

    def freeze_feature_extractor(self):
        for name, param in self.pretrained_encoder.named_parameters():
            if "regressor" not in name:
                param.requires_grad = False




if __name__ == "__main__":
    model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
    print(model.encoder.layer[1].attention.attention.query.weight.mean())

    config = VideoMAEConfig()
    model = VideoMAEModel(config)
    print(model.encoder.layer[1].attention.attention.query.weight.mean())

