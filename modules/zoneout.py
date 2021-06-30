import torch
import torch.nn as nn


class ZoneoutLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, zoneout_prob=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.zoneout_prob = zoneout_prob
        self.lstm = nn.LSTMCell(input_size, hidden_size, bias)
        self.dropout = nn.Dropout(p=zoneout_prob)

        # initialize all forget gate bias of LSTM to 1.0
        # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745/4
        self.lstm.bias_ih[hidden_size:2*hidden_size].data.fill_(1.0)
        self.lstm.bias_hh[hidden_size:2*hidden_size].data.fill_(1.0)

    def forward(self, x, prev_hc=None):
        h, c = self.lstm(x, prev_hc)

        if prev_hc is None:
            prev_h = torch.zeros(x.size(0), self.hidden_size)
            prev_c = torch.zeros(x.size(0), self.hidden_size)
        else:
            prev_h, prev_c = prev_hc

        if self.training:
            h = (1. - self.zoneout_prob) * self.dropout(h - prev_h) + prev_h
            c = (1. - self.zoneout_prob) * self.dropout(c - prev_c) + prev_c
        else:
            h = (1. - self.zoneout_prob) * h + self.zoneout_prob * prev_h
            c = (1. - self.zoneout_prob) * c + self.zoneout_prob * prev_c

        return h, c
