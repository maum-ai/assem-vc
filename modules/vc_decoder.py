import torch
import torch.nn as nn
import torch.nn.functional as F

from .cond_bn import ConditionalBatchNorm1d


# adopted Generator ResBlock from https://arxiv.org/abs/1909.11646
class GBlock(nn.Module):
    def __init__(self, in_channels, out_channels, condition_dim, dropout):
        super().__init__()
        self.cond_bn = nn.ModuleList([
            ConditionalBatchNorm1d(in_channels if i==0 else out_channels, condition_dim)
            for i in range(4)])
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        self.cnn = nn.ModuleList([
            nn.Conv1d(in_channels if i==0 else out_channels, out_channels,
                kernel_size=3, dilation=2**i, padding=2**i)
            for i in range(4)])
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, z, mask=None):
        identity = x
        x = self.cnn[0](self.dropout(self.leaky_relu(self.cond_bn[0](x, z))))
        
        if mask is not None:
            x.masked_fill_(mask, 0.0)

        x = self.cnn[1](self.dropout(self.leaky_relu(self.cond_bn[1](x, z))))
        if mask is not None:
            x.masked_fill_(mask, 0.0)

        x = x + self.shortcut(identity)
        if mask is not None:
            x.masked_fill_(mask, 0.0)

        identity = x
        x = self.cnn[2](self.dropout(self.leaky_relu(self.cond_bn[2](x, z))))
        if mask is not None:
            x.masked_fill_(mask, 0.0)

        x = self.cnn[3](self.dropout(self.leaky_relu(self.cond_bn[3](x, z))))
        if mask is not None:
            x.masked_fill_(mask, 0.0)

        x = x + identity
        return x


class VCDecoder(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.stem = nn.Conv1d(hp.chn.encoder + hp.chn.residual_out, hp.chn.gblock[0], kernel_size=7, padding=3)
        self.gblock = nn.ModuleList([
            GBlock(in_channels, out_channels, hp.chn.speaker.token, hp.train.dropout)
            for in_channels, out_channels in
            zip(list(hp.chn.gblock)[:-1], hp.chn.gblock[1:])])
        self.final = nn.Conv1d(hp.chn.gblock[-1], hp.audio.n_mel_channels, kernel_size=1)

    def forward(self, x, speaker_emb, mask=None):
        # x: linguistic features + pitch info.
        # [B, chn.encoder + chn.residual_out, T_dec]
        x = self.stem(x)  # [B, chn.gblock[0], T]
        if mask is not None:
            x.masked_fill_(mask, 0.0)

        for gblock in self.gblock:
            x = gblock(x, speaker_emb, mask)
        # x: [B, chn.gblock[-1], T]

        x = self.final(x)  # [B, M, T]
        if mask is not None:
            x.masked_fill_(mask, 0.0)
        return x
