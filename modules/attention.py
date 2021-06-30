import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import gamma


class StaticFilter(nn.Module):
    def __init__(self, channels, kernel_size, out_dim):
        super().__init__()
        assert kernel_size % 2 == 1, \
            'kernel size of StaticFilter must be odd, got %d' % kernel_size
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv1d(1, channels, kernel_size=kernel_size, padding=padding)
        self.fc = nn.Linear(channels, out_dim, bias=False)

    def forward(self, prev_attn):
        # prev_attn: [B, T]
        x = prev_attn.unsqueeze(1)  # [B, 1, T]
        x = self.conv(x)  # [B, channels, T]
        x = x.transpose(1, 2)  # [B, T, out_dim]
        x = self.fc(x)
        return x


class DynamicFilter(nn.Module):
    def __init__(self, channels, kernel_size, attn_rnn_dim, hypernet_dim, out_dim):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        assert kernel_size % 2 == 1, \
            'kernel size of DynamicFilter must be odd, god %d' % kernel_size
        self.padding = (kernel_size - 1) // 2

        self.hypernet = nn.Sequential(
            nn.Linear(attn_rnn_dim, hypernet_dim),
            nn.Tanh(),
            nn.Linear(hypernet_dim, channels*kernel_size),
        )
        self.fc = nn.Linear(channels, out_dim)

    def forward(self, query, prev_attn):
        # query: [B, attn_rnn_dim]
        # prev_attn: [B, T]
        B, T = prev_attn.shape
        convweight = self.hypernet(query)  # [B, channels * kernel_size]
        convweight = convweight.view(B, self.channels, self.kernel_size)
        convweight = convweight.view(B * self.channels, 1, self.kernel_size)
        prev_attn = prev_attn.unsqueeze(0)
        x = F.conv1d(prev_attn, convweight, padding=self.padding, groups=B)
        x = x.view(B, self.channels, T)
        x = x.transpose(1, 2)  # [B, T, channels]
        x = self.fc(x)  # [B, T, out_dim]
        return x


class PriorFilter(nn.Module):
    def __init__(self, causal_n, alpha, beta):
        super().__init__()
        self.causal_n = causal_n
        self.alpha = alpha
        self.beta = beta

        def beta_func(x, y):
            return gamma(x) * gamma(y) / gamma(x+y)

        def p(n, k, alpha, beta):
            def nCr(n, r):
                f = math.factorial
                return f(n) / (f(r) * f(n-r))
            return nCr(n, k) * beta_func(k+alpha, n-k+beta) / beta_func(alpha, beta)

        self.prior = np.array([
            p(self.causal_n-1, i, self.alpha, self.beta)
            for i in range(self.causal_n)[::-1]]).astype(np.float32)

        self.prior = torch.from_numpy(self.prior)
        self.prior = self.prior.view(1, 1, -1)
        self.register_buffer('prior_filter', self.prior)

    def forward(self, prev_attn):
        prev_attn = prev_attn.unsqueeze(1)
        energies = F.conv1d(F.pad(prev_attn, (self.causal_n-1, 0)), self.prior_filter)
        energies = energies.squeeze(1)
        energies = torch.log(torch.clamp(energies, min=1e-8))
        return energies


class Attention(nn.Module):
    def __init__(self, attn_rnn_dim, attn_dim, static_channels, static_kernel_size,
                dynamic_channels, dynamic_kernel_size, causal_n, causal_alpha, causal_beta):
        super().__init__()
        self.v = nn.Linear(attn_dim, 1, bias=False)
        self.static_filter = StaticFilter(static_channels, static_kernel_size, attn_dim)
        self.dynamic_filter = DynamicFilter(dynamic_channels, dynamic_kernel_size, attn_rnn_dim, attn_dim, attn_dim)
        self.prior_filter = PriorFilter(causal_n, causal_alpha, causal_beta)
        self.score_mask_value = -float('inf')

    def get_alignment_energies(self, query, prev_attn):
        static_result = self.static_filter(prev_attn)
        dynamic_result = self.dynamic_filter(query, prev_attn)
        prior_result = self.prior_filter(prev_attn)

        energies = self.v(torch.tanh(static_result + dynamic_result)).squeeze(-1) + prior_result
        return energies

    def forward(self, attn_hidden, memory, prev_attn, mask):
        alignment = self.get_alignment_energies(attn_hidden, prev_attn)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attn_weights = F.softmax(alignment, dim=1)  # [B, T]
        context = torch.bmm(attn_weights.unsqueeze(1), memory)
        # [B, 1, T] @ [B, T, (chn.encoder + chn.speaker)] -> [B, 1, (chn.encoder + chn.speaker)]
        context = context.squeeze(1)

        return context, attn_weights
