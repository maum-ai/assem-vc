import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    def __init__(self, channels, kernel_size, depth, n_symbols):
        super().__init__()
        self.embedding = nn.Embedding(n_symbols, channels)

        padding = (kernel_size - 1) // 2
        self.cnn = list()
        for _ in range(depth):
            self.cnn.append(nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(channels),
                nn.ReLU(),
                nn.Dropout(0.5),
            ))
        self.cnn = nn.Sequential(*self.cnn)

        self.lstm = nn.LSTM(channels, channels//2, 1, batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        x = self.embedding(x)  # [B, T, emb]
        x = x.transpose(1, 2)  # [B, emb, T]
        x = self.cnn(x)  # [B, chn, T]
        x = x.transpose(1, 2)  # [B, T, chn]

        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(
            x, batch_first=True)

        return x

    def inference(self, x):
        x = self.embedding(x)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)
        return x


class SpeakerEncoder(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.relu = nn.ReLU()
        self.stem = nn.Conv2d(
            1, hp.chn.speaker.cnn[0], kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))
        self.cnn = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))
            for in_channels, out_channels in zip(list(hp.chn.speaker.cnn)[:-1], hp.chn.speaker.cnn[1:])
        ]) # 80 - 40 - 20 - 10 - 5 - 3 - 2
        self.bn = nn.ModuleList([
            nn.BatchNorm2d(channels) for channels in hp.chn.speaker.cnn
        ])
        self.gru = nn.GRU(hp.chn.speaker.cnn[-1]*2, hp.chn.speaker.token,
                          batch_first=True, bidirectional=False)

    def forward(self, x, input_lengths):
        # x: [B, mel, T]
        x = x.unsqueeze(1)  # [B, 1, mel, T]
        x = self.stem(x)
        input_lengths = (input_lengths + 1) // 2

        for cnn, bn in zip(self.cnn, self.bn):
            x = bn(x)
            x = self.relu(x)
            x = cnn(x)
            input_lengths = (input_lengths + 1) // 2

        x = x.view(x.size(0), -1, x.size(-1))  # [B, chn.speaker.cnn[-1]*2, T}]
        x = x.transpose(1, 2)  # [B, T, chn.speaker.cnn[-1]*2]

        input_lengths, indices = torch.sort(input_lengths, descending=True)
        x = torch.index_select(x, dim=0, index=indices)

        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.gru.flatten_parameters()
        _, x = self.gru(x)

        x = torch.index_select(x[0], dim=0, index=torch.sort(indices)[1])
        return x

    def inference(self, x):
        x = x.unsqueeze(1)
        x = self.stem(x)

        for cnn, bn in zip(self.cnn, self.bn):
            x = bn(x)
            x = self.relu(x)
            x = cnn(x)

        x = x.view(x.size(0), -1, x.size(-1))
        x = x.transpose(1, 2)

        self.gru.flatten_parameters()
        _, x = self.gru(x)
        x = x.squeeze(1)
        return x
