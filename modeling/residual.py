import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualLayer2D(nn.Module):
    def __init__(self, in_dim, res_dim):
        """
        - in_dim: the input dimension
        - res_dim: the intermediate dimension
        """
        super(ResidualLayer2D, self).__init__()

        self.residual_block = nn.Sequential(
            nn.Conv2d(in_dim, res_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(res_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(res_dim, res_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(res_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(res_dim, in_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(in_dim),
        )

    def forward(self, x):
        return F.relu(x + self.residual_block(x))

class ResidualLayer3D(nn.Module):
    def __init__(self, in_dim, res_dim):
        """
        - in_dim: the input dimension
        - res_dim: the intermediate dimension
        """
        super(ResidualLayer3D, self).__init__()

        self.residual_block = nn.Sequential(
            nn.Conv3d(in_dim, res_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(res_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(res_dim, res_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(res_dim),
            nn.ReLU(inplace=True),
            nn.Conv3d(res_dim, in_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm3d(in_dim),
        )

    def forward(self, x):
        return F.relu(x + self.residual_block(x))


class ResidualStack(nn.Module):
    
    def __init__(self, in_dim, res_dim, num_reslayers=3, mode="2D"):
        """
        - in_dim: the input dimension
        - res_dim: the intermediate dimension
        - num_reslayers: the number of residual layers
        - mode: 2D or 3D
        """
        super(ResidualStack, self).__init__()

        if mode == "2D":
            self.residual_stack = nn.Sequential(
                *[ResidualLayer2D(in_dim, res_dim)] * num_reslayers
            )
        else:
            self.residual_stack = nn.Sequential(
                *[ResidualLayer3D(in_dim, res_dim)] * num_reslayers
            )
    
    def forward(self, x):
        for res_layer in self.residual_stack:
            x = res_layer(x)
        return x


if __name__ == "__main__":

    B = 20
    H = 32
    W = 32
    C = 768
    x_dummy = torch.randn(B, C, H, W)

    residual_layer = ResidualLayer2D(C, 2 * C)
    y_dummy = residual_layer(x_dummy)
    

    residual_stack = ResidualStack(C, 2*C, 5, mode="2D")
    y_dummy = residual_stack(x_dummy)
    print(x_dummy[0, 1, 1, 0], x_dummy.size())
    print(y_dummy[0, 1, 1, 0], y_dummy.size())



