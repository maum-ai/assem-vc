import os
import re
import torch
import random
import librosa
import numpy as np
from pysptk import sptk
from torch.utils.data import Dataset
from collections import Counter

from .text import Language
from .text.cmudict import CMUDict
from modules.mel import mel_spectrogram


class TextMelDataset(Dataset):
    def __init__(self, hp, data_dir, metadata_path, train=True, norm=False, use_f0s = False):
        super().__init__()
        self.hp = hp
        self.lang = Language(hp.data.lang, hp.data.text_cleaners)
        self.train = train
        self.norm = norm
        self.data_dir = data_dir
        metadata_path = os.path.join(data_dir, metadata_path)
        self.meta = self.load_metadata(metadata_path)
        self.use_f0s = use_f0s
        if use_f0s:
            f0s_path = hp.data.f0s_list_path
            self.f0s_means, self.f0s_vars = self.load_f0s_lists(f0s_path)

        self.speaker_dict = {speaker: idx for idx, speaker in enumerate(hp.data.speakers)}
        self.mel_fmin = hp.audio.mel_fmin
        self.f0_min = hp.audio.f0_min
        self.f0_max = hp.audio.f0_max
        self.harm_thresh = hp.audio.harm_thresh

        if train:
            # balanced sampling for each speaker
            speaker_counter = Counter((spk_id \
                for audiopath, text, spk_id in self.meta))
            weights = [1.0 / speaker_counter[spk_id] \
                for audiopath, text, spk_id in self.meta]
            self.mapping_weights = torch.DoubleTensor(weights)

        if hp.data.lang == 'eng2':
            self.cmudict = CMUDict(hp.data.cmudict_path)
            self.cmu_pattern = re.compile(r'^(?P<word>[^!\'(),-.:~?]+)(?P<punc>[!\'(),-.:~?]+)$')
        else:
            self.cmudict = None

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if self.train:
            idx = torch.multinomial(self.mapping_weights, 1).item()

        audiopath, text, spk_id = self.meta[idx]

        audiopath = os.path.join(self.data_dir, audiopath)
        text_norm = self.get_text(text)
        new_path = '{}.gta'.format(audiopath)

        if spk_id not in self.hp.data.speakers:
            mel, f0 = self.get_mel_and_f0(audiopath)
            return text_norm, mel, 0, 0, f0

        spk_id = self.speaker_dict[spk_id]

        if self.use_f0s:
            f0_mean, f0_var = self.f0s_means[spk_id], self.f0s_vars[spk_id]
            mel, f0 = self.get_mel_and_f0(audiopath, f0_mean, f0_var)

            return text_norm, mel, spk_id, new_path, f0

        mel = self.get_mel_and_f0(audiopath)

        return text_norm, mel, spk_id, new_path

    def get_f0(self, audio, f0_mean=None, f0_var=None, sampling_rate=22050, frame_length=1024,
               hop_length=256, f0_min=80, f0_max=880, harm_thresh=0.25, mel_fmin = 70.0):

        '''f0, harmonic_rates, argmins, times = compute_yin(
            audio, sampling_rate, frame_length, hop_length, f0_min, f0_max,
            harm_thresh, mel_fmin)'''
        f0 = sptk.rapt(audio*32768, sampling_rate, hop_length, min=f0_min, max=f0_max, otype=2)

        f0 = np.clip(f0, 0, f0_max)

        index_nonzero = np.nonzero(f0)
        f0[index_nonzero] += 10.0
        f0 -= 10.0

        if f0_mean == None:
            f0_mean =  np.mean(f0[index_nonzero])
        if f0_var == None:
            f0_var =  np.std(f0[index_nonzero])

        f0[index_nonzero] = (f0[index_nonzero] - f0_mean) / f0_var

        return f0

    def get_mel_and_f0(self, audiopath, f0_mean=None, f0_var=None):
        wav, sr = librosa.load(audiopath, sr=None, mono=True)
        assert sr == self.hp.audio.sampling_rate, \
            'sample mismatch: expected %d, got %d at %s' % (self.hp.audio.sampling_rate, sr, audiopath)
        wav = torch.from_numpy(wav)
        wav = wav.unsqueeze(0)

        if self.norm:
            wav = wav * (0.99 / (torch.max(torch.abs(wav)) + 1e-7))

        mel = mel_spectrogram(wav, self.hp.audio.filter_length, self.hp.audio.n_mel_channels,
                              self.hp.audio.sampling_rate,
                              self.hp.audio.hop_length, self.hp.audio.win_length,
                              self.hp.audio.mel_fmin, self.hp.audio.mel_fmax, center=False)
        mel = mel.squeeze(0)
        wav = wav.cpu().numpy()[0]

        if self.use_f0s:
            f0 = self.get_f0(wav, f0_mean, f0_var, self.hp.audio.sampling_rate,
                             self.hp.audio.filter_length, self.hp.audio.hop_length, self.f0_min,
                             self.f0_max, self.harm_thresh, self.mel_fmin)
            f0 = torch.from_numpy(f0)[None]
            f0 = f0[:, :mel.size(1)]

            return mel, f0

        return mel

    def get_text(self, text):
        if self.cmudict is not None:
            text = ' '.join([self.get_arpabet(word) for word in text.split(' ')])
        text_norm = torch.IntTensor(self.lang.text_to_sequence(text, self.hp.data.text_cleaners))
        return text_norm

    def get_arpabet(self, word):
        arpabet = self.cmudict.lookup(word)
        if arpabet is None:
            match = self.cmu_pattern.search(word)
            if match is None:
                return word
            subword = match.group('word')
            arpabet = self.cmudict.lookup(subword)
            if arpabet is None:
                return word
            punc = match.group('punc')
            arpabet = '{%s}%s' % (arpabet[0], punc)
        else:
            arpabet = '{%s}' % arpabet[0]

        if random.random() < 0.5:
            return word
        else:
            return arpabet

    def load_metadata(self, path, split="|"):
        with open(path, 'r', encoding='utf-8') as f:
            metadata = [line.strip().split(split) for line in f]

        return metadata

    def load_f0s_lists(self, path, split="|"):

        with open(path, 'r', encoding='utf-8') as f:
            metadata = [line.strip().split(split) for line in f]

        f0s_spk_list = [line[0] for line in metadata]
        assert f0s_spk_list == self.hp.data.speakers, \
            'speaker list mismatch in f0s_list: expected %s, but %s at %s' \
            % (self.hp.data.speakers, f0s_spk_list, path)
        f0s_mean_list = [float(line[1]) for line in metadata]
        f0s_var_list = [float(line[2]) for line in metadata]

        return f0s_mean_list, f0s_var_list


def text_mel_collate(batch):
    input_lengths, ids_sorted_decreasing = torch.sort(
        torch.LongTensor([len(x[0]) for x in batch]),
        dim=0, descending=True)
    max_input_len = torch.empty(len(batch), dtype=torch.long)
    max_input_len.fill_(input_lengths[0])

    text_padded = torch.zeros((len(batch), max_input_len[0]), dtype=torch.long)
    n_mel_channels = batch[0][1].size(0)
    max_target_len = max([x[1].size(1) for x in batch])
    mel_padded = torch.zeros(len(batch), n_mel_channels, max_target_len)
    output_lengths = torch.empty(len(batch), dtype=torch.long)
    speakers = torch.empty(len(batch), dtype=torch.long)
    new_paths = []

    use_f0s = False
    if len(batch[0]) == 5:
        use_f0s = True
        f0_padded = torch.FloatTensor(len(batch), 1, max_target_len)
        f0_padded.zero_()

    for idx, key in enumerate(ids_sorted_decreasing):
        text = batch[key][0]
        text_padded[idx, :text.size(0)] = text
        mel = batch[key][1]
        mel_padded[idx, :, :mel.size(1)] = mel
        output_lengths[idx] = mel.size(1)
        speakers[idx] = batch[key][2]
        new_paths.append(batch[key][3])
        if use_f0s:
            f0 = batch[key][4]
            f0_padded[idx, :, :f0.size(1)] = f0

    if use_f0s:
        return text_padded, mel_padded, speakers, f0_padded, \
            input_lengths, output_lengths, max_input_len, new_paths

    return text_padded, mel_padded, speakers, \
           input_lengths, output_lengths, max_input_len, new_paths


