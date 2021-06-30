import torch
import torch.nn as nn
import torch.nn.functional as F


# adopted from https://github.com/pytorch/pytorch/issues/8985#issuecomment-405080775
class ConditionalBatchNorm1d(nn.Module):
    def __init__(self, num_features, condition_dim):
        super().__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm1d(num_features, affine=False)
        self.projection = nn.Linear(condition_dim, 2*num_features)

    def forward(self, x, cond):
        # x: [B, num_features, T]
        # cond: [B, condition_dim]
        x = self.bn(x)
        gamma, beta = self.projection(cond).chunk(2, dim=1)
        x = gamma.unsqueeze(-1) * x + beta.unsqueeze(-1)
        return x
