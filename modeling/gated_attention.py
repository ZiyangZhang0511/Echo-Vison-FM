import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedAttention(nn.Module):
    def __init__(self, feat_dim, L=512):
        super(GatedAttention, self).__init__()

        self.V = nn.parameter.Parameter(torch.zeros((L, feat_dim)))
        self.U = nn.parameter.Parameter(torch.zeros((L, feat_dim)))
        self.W = nn.parameter.Parameter(torch.zeros((L, 1)))

        torch.nn.init.kaiming_uniform_(self.V.data)
        torch.nn.init.kaiming_uniform_(self.U.data)
        torch.nn.init.kaiming_uniform_(self.W.data)


    def forward(self, x):
        tmp1 = F.tanh(torch.matmul(x, self.V.T))
        tmp2 = F.sigmoid(torch.matmul(x, self.U.T))

        attention_scores = torch.matmul((tmp1 * tmp2), self.W)

        attention_weights = F.softmax(attention_scores, dim=1)

        x_merged = torch.sum(x * attention_weights, dim=1)

        return x_merged


if __name__ == "__main__":

    x = torch.randn((4, 1568, 768))

    model = GatedAttention(768)

    y = model(x)

    # for param in model.parameters():
    #     print(param, param.requires_grad)
    print(y.size())
    # print(y.sum(dim=1))