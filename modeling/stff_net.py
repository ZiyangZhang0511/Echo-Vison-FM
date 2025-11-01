import torch
import torch.nn as nn
import torch.nn.functional as F

from .residual import ResidualStack
from .gated_attention import GatedAttention

from positional_encodings.torch_encodings import (
    PositionalEncoding1D, 
    PositionalEncoding2D, 
    PositionalEncoding3D, 
    Summer,
)

import numpy as np

def get_1D_positional_encodings(indices, d_model):

    def get_angles(pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    indices = np.array(indices.detach().cpu().clone())
    
    angle_rads = get_angles(indices[:, :, np.newaxis], np.arange(d_model)[np.newaxis, np.newaxis, :], d_model)

    angle_rads[:, :, 0::2] = np.sin(angle_rads[:, :, 0::2])
    angle_rads[:, :, 1::2] = np.cos(angle_rads[:, :, 1::2])

    return torch.from_numpy(angle_rads).to(torch.float32)

class SpatioTemporalFeatureFusionNet(nn.Module):

    def __init__(self, feat_dim, size_feat_map):
        super(SpatioTemporalFeatureFusionNet, self).__init__()

        self.feat_dim = feat_dim
        self.size_feat_map = size_feat_map

        self.spatial_temporal_positional_encoding = PositionalEncoding3D(feat_dim)
        self.residual_stack_all = ResidualStack(feat_dim, feat_dim//2, num_reslayers=1, mode="3D")
        self.gated_attention_all = GatedAttention(feat_dim, L=1024)

        self.residual_stack_spatial = ResidualStack(feat_dim, feat_dim//2, num_reslayers=1, mode="2D")
        self.conv_spatial = nn.Conv2d(feat_dim, feat_dim, kernel_size=(14, 14), stride=1, padding=0)
        self.gated_attention_temporal = GatedAttention(feat_dim, L=1024)
        self.fc_pe = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, self.feat_dim),
        )
    
    def forward(self, x, temporal_indices):
        bs = x.size(0)
        T = self.size_feat_map[0]
        H = self.size_feat_map[1]
        W = self.size_feat_map[2]

        x = x.view(-1, T, H, W, self.feat_dim)

        ###========joint spatiotemporal========###

        x_3dpe = self.spatial_temporal_positional_encoding(x) + x
        x_3dpe = x_3dpe.permute(0, 4, 1, 2, 3).contiguous()
        x_3dpe_res = self.residual_stack_all(x_3dpe)

        x_seq_all = x_3dpe_res.view(-1, T*H*W, self.feat_dim)

        global_embedding = self.gated_attention_all(x_seq_all)
        # print(global_embedding.size())

        ###========spatial-temporal seperation========###
        x_spatial = x.view(bs*T, self.feat_dim, H, W)
        # print(x_spatial.size())
        x_spatial = self.residual_stack_spatial(x_spatial)
        x_spatial = self.conv_spatial(x_spatial)
        # print(x_spatial.size())
        x_temporal = x_spatial.view(bs, T, self.feat_dim)
        # print(x_temporal.size())

        sparse_pe = get_1D_positional_encodings(temporal_indices, self.feat_dim).to(x.device)
        # print(sparse_pe.size())
        sparse_pe_ave = sparse_pe.view(-1, temporal_indices.shape[1]//2, 2, self.feat_dim).mean(dim=2)
        # print(sparse_pe_ave.size(), sparse_pe_ave.min())
        sparse_pe_transformed = self.fc_pe(sparse_pe_ave.view(-1, self.feat_dim))
        sparse_pe_transformed = sparse_pe_transformed.view(-1, temporal_indices.shape[1]//2, self.feat_dim)
        # print(sparse_pe_transformed.size(), sparse_pe_transformed.min())

        x_temporal = x_temporal + sparse_pe_transformed
        # print(x_temporal.size())

        fine_embeddings = self.gated_attention_temporal(x_temporal)
        # print(fine_embeddings.size())

        stf_embeddings = torch.cat([global_embedding, fine_embeddings], dim=1)

        return stf_embeddings


if __name__ == "__main__":
    feat_dim = 768
    size_feat_map = (8, 14, 14)
    num_seq = 8 * 14 * 14
    bs = 40
    temporal_indices = torch.tensor([0, 13, 15, 17, 19, 40, 41, 42, 50, 80, 100, 123, 145, 156, 166, 200], dtype=torch.int32).to("cuda")
    temporal_indices = temporal_indices.unsqueeze(dim=0).repeat(bs, 1)
    print(temporal_indices.size())
    x = torch.randn((bs, num_seq, feat_dim)).to("cuda")

    stf_net = SpatioTemporalFeatureFusionNet(feat_dim, size_feat_map).to("cuda")

    re = stf_net(x, temporal_indices)
    print(re.size(), re.device)

    # pe = torch.from_numpy(get_1D_positional_encodings(temporal_indices, feat_dim))
    # print(pe.size())



