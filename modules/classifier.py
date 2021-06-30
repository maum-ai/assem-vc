import torch
import torch.nn as nn
import torch.nn.functional as F


class SpkClassifier(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hp.chn.speaker.token, hp.chn.speaker.token),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hp.chn.speaker.token, len(hp.data.speakers))
        )

    def forward(self, x):
        x = self.mlp(x)
        x = F.log_softmax(x, dim=1)
        return x
