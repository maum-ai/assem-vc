import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention
from .zoneout import ZoneoutLSTMCell


class PreNet(nn.Module):
    def __init__(self, channels, in_dim, depth):
        super().__init__()
        sizes = [in_dim] + [channels] * depth
        self.layers = nn.ModuleList(
            [nn.Linear(in_size, out_size)
             for (in_size, out_size) in zip(sizes[:-1], sizes[1:])])

    # in default tacotron2 setting, we use prenet_dropout=0.5 for both train/infer.
    # you may want to set prenet_dropout=0.0 for some case.
    def forward(self, x, prenet_dropout):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=prenet_dropout, training=True)
        return x


class PostNet(nn.Module):
    def __init__(self, channels, kernel_size, n_mel_channels, depth):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.cnn = list()
        self.cnn.append(
            nn.Sequential(
                nn.Conv1d(n_mel_channels, channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(channels),
                nn.Tanh(),
                nn.Dropout(0.5),))

        for i in range(1, depth - 1):
            self.cnn.append(
                nn.Sequential(
                    nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm1d(channels),
                    nn.Tanh(),
                    nn.Dropout(0.5),))

        self.cnn.append(
            nn.Sequential(
                nn.Conv1d(channels, n_mel_channels, kernel_size=kernel_size, padding=padding),))

        self.cnn = nn.Sequential(*self.cnn)

    def forward(self, x):
        return self.cnn(x)


class TTSDecoder(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.go_frame = nn.Parameter(
            torch.randn(1, hp.audio.n_mel_channels), requires_grad=True)

        self.prenet = PreNet(
            hp.chn.prenet, in_dim=hp.audio.n_mel_channels, depth=hp.depth.prenet)
        self.postnet = PostNet(
            hp.chn.postnet, hp.ker.postnet, hp.audio.n_mel_channels, hp.depth.postnet)
        self.attention_rnn = ZoneoutLSTMCell(
            hp.chn.prenet + hp.chn.encoder + hp.chn.speaker.token, hp.chn.attention_rnn, zoneout_prob=0.1)
        self.attention_layer = Attention(
            hp.chn.attention_rnn, hp.chn.attention, hp.chn.static, hp.ker.static,
            hp.chn.dynamic, hp.ker.dynamic, hp.ker.causal, hp.ker.alpha, hp.ker.beta)
        self.decoder_rnn = ZoneoutLSTMCell(
            hp.chn.attention_rnn + hp.chn.encoder + hp.chn.speaker.token, hp.chn.decoder_rnn, zoneout_prob=0.1)
        self.mel_fc = nn.Linear(
            hp.chn.decoder_rnn + hp.chn.encoder + hp.chn.speaker.token, hp.audio.n_mel_channels)

    def get_go_frame(self, memory):
        return self.go_frame.expand(memory.size(0), self.hp.audio.n_mel_channels)

    def initialize(self, memory, mask):
        B, T, _ = memory.size()
        self.memory = memory
        self.mask = mask
        device = memory.device

        attn_h = torch.zeros(B, self.hp.chn.attention_rnn).to(device)
        attn_c = torch.zeros(B, self.hp.chn.attention_rnn).to(device)
        dec_h = torch.zeros(B, self.hp.chn.decoder_rnn).to(device)
        dec_c = torch.zeros(B, self.hp.chn.decoder_rnn).to(device)

        prev_attn = torch.zeros(B, T).to(device)
        prev_attn[:, 0] = 1.0
        context = torch.zeros(B, self.hp.chn.encoder + self.hp.chn.speaker.token).to(device)

        return attn_h, attn_c, dec_h, dec_c, prev_attn, context

    def decode(self, x, attn_h, attn_c, dec_h, dec_c, prev_attn, context):
        x = torch.cat((x, context), dim=-1)
        # [B, chn.prenet + (chn.encoder + chn.speaker.token)]
        attn_h, attn_c = self.attention_rnn(x, (attn_h, attn_c))
        # [B, chn.attention_rnn]
        
        context, prev_attn = self.attention_layer(attn_h, self.memory, prev_attn, self.mask)
        # context: [B, (chn.encoder + chn.speaker.token)], prev_attn: [B, T]

        x = torch.cat((attn_h, context), dim=-1) 
        # [B, chn.attention_rnn + (chn.encoder + chn.speaker.token)]
        dec_h, dec_c = self.decoder_rnn(x, (dec_h, dec_c))
        # [B, chn.decoder_rnn]

        x = torch.cat((dec_h, context), dim=-1)
        # [B, chn.decoder_rnn + (chn.encoder + chn.speaker.token)]
        mel_out = self.mel_fc(x)
        # [B, audio.n_mel_channels]

        return mel_out, attn_h, attn_c, dec_h, dec_c, prev_attn, context

    def parse_decoder_outputs(self, mel_outputs, alignments):
        # 'T' is T_dec.
        mel_outputs = torch.stack(mel_outputs, dim=0).transpose(0, 1).contiguous()
        mel_outputs = mel_outputs.transpose(1, 2)
        # mel: [T, B, M] -> [B, T, M] -> [B, M, T]
        alignments = torch.stack(alignments, dim=0).transpose(0, 1).contiguous()
        # align: [T_dec, B, T_enc] -> [B, T_dec, T_enc]

        return mel_outputs, alignments

    def forward(self, x, memory, memory_lengths, output_lengths, max_input_len,
                prenet_dropout=0.5, no_mask=False, tfrate=True):
        # x: mel spectrogram for teacher-forcing. [B, M, T].
        go_frame = self.get_go_frame(memory).unsqueeze(0)
        x = x.transpose(1, 2).transpose(0, 1)  # [B, M, T] -> [B, T, M] -> [T, B, M]
        x = torch.cat((go_frame, x), dim=0) # [T+1, B, M]
        x = self.prenet(x, prenet_dropout)

        attn_h, attn_c, dec_h, dec_c, prev_attn, context = \
            self.initialize(memory,
                mask=None if no_mask else ~self.get_mask_from_lengths(memory_lengths))
        mel_outputs, alignments = [], []

        decoder_input = x[0]
        while len(mel_outputs) < x.size(0) - 1:
            mel_out, attn_h, attn_c, dec_h, dec_c, prev_attn, context = \
                self.decode(decoder_input, attn_h, attn_c, dec_h, dec_c, prev_attn, context)

            mel_outputs.append(mel_out)
            alignments.append(prev_attn)

            if tfrate and self.hp.train.teacher_force.rate < random.random():
                decoder_input = self.prenet(mel_out, prenet_dropout)
            else:
                decoder_input = x[len(mel_outputs)]

        mel_outputs, alignments = self.parse_decoder_outputs(mel_outputs, alignments)
        mel_postnet = mel_outputs + self.postnet(mel_outputs)

        # DataParallel expects equal sized inputs/outputs, hence padding
        alignments = alignments.unsqueeze(0)
        alignments = F.pad(alignments, (0, max_input_len[0] - alignments.size(-1)), 'constant', 0)
        alignments = alignments.squeeze(0)
        
        mel_outputs, mel_postnet, alignments = \
            self.mask_output(mel_outputs, mel_postnet, alignments, output_lengths)
        return mel_outputs, mel_postnet, alignments

    def get_mask_from_lengths(self, lengths, max_len=None):
        if max_len is None:
            max_len = torch.max(lengths).item()
        ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
        mask = (ids < lengths.unsqueeze(1))
        return mask

    def mask_output(self, mel_outputs, mel_postnet, alignments, output_lengths=None):
        if self.hp.train.mask_padding and output_lengths is not None:
            mask = ~self.get_mask_from_lengths(output_lengths, max_len=mel_outputs.size(-1))
            mask = mask.unsqueeze(1)  # [B, 1, T] torch.bool
            mel_outputs.masked_fill_(mask, 0.0)
            mel_postnet.masked_fill_(mask, 0.0)

        return mel_outputs, mel_postnet, alignments
