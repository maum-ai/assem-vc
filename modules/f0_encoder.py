import torch.nn as nn

class ConvNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)
        self.conv = nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)
        nn.init.xavier_uniform_(
            self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal

class F0_Encoder(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.prenet_f0 = ConvNorm(
            1, hp.chn.prenet_f0,
            kernel_size=hp.ker.prenet_f0,
            padding=max(0, int(hp.ker.prenet_f0 / 2)),
            bias=True, stride=1, dilation=1)

    def forward(self, f0s):
        f0s = self.prenet_f0(f0s)
        return f0s
