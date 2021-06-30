from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
from omegaconf import OmegaConf
import os
import tqdm
from scipy.io.wavfile import write, read
import torch
import glob
from pysptk import sptk
from random import shuffle
import math
from argparse import ArgumentParser

def get_f0(audio, sampling_rate, frame_length,
           hop_length, f0_min, f0_max, harm_thresh):
    f0 = sptk.rapt(audio * 32768, sampling_rate, hop_length, min=f0_min, max=f0_max, otype=2)
    f0 = np.clip(f0, 0, f0_max)
    return f0


def _process_speaker(filepath, sampling_rate, frame_length,
                     hop_length, f0_min, f0_max, harm_thresh):
    audio, sr = load_wav_to_torch(filepath)

    assert sr == sampling_rate, \
        'sample mismatch: expected %d, got %d at %s' % (sampling_rate, sr, filepath)

    f0 = get_f0(audio.cpu().numpy(), sampling_rate, frame_length,
                     hop_length, f0_min, f0_max, harm_thresh)
    f0 = f0[np.nonzero(f0)]
    f0_sq =np.square(f0)

    square_over_frames = np.sum(f0_sq)
    sum_over_frames = np.sum(f0)
    n_frames = len(f0)

    return square_over_frames, sum_over_frames, n_frames


def write_metadata(metadata, out_file):
    with open(out_file, 'w', encoding='utf-8') as f:
        for m in metadata:
            if m is None:
                continue
            f.write('|'.join([str(x) for x in m]) + '\n')


def load_wav_to_torch(full_path):
    # scipy.wavefil.read does not take care of the case where wav is int or uint.
    # Thus, scipy.read is replaced with read_wav_np
    sampling_rate, data = read_wav_np(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def read_wav_np(path):
    try:
        sr, wav = read(path)
    except Exception as e:
        print(str(e) + path)
        return Exception

    if len(wav.shape) == 2:
        wav = wav[:, 0]

    if wav.dtype == np.int16:
        wav = wav / 32768.0
    elif wav.dtype == np.int32:
        wav = wav / 2147483648.0
    elif wav.dtype == np.uint8:
        wav = (wav - 128) / 128.0

    wav = wav.astype(np.float32)

    return sr, wav


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="path of configuration yaml file")
    parser.add_argument('--num_workers', type=int, default=32,
                        help="number of workers")
    parser.add_argument('-o', '--output_filename', type=str, default='f0s.txt',
                        help="name of the output file")
    args = parser.parse_args()

    hp = OmegaConf.load(args.config)
    with open(os.path.join(hp.data.train_dir, hp.data.train_meta), 'r', encoding='utf-8') as g:
        data = g.readlines()
    wavdir = [x.split('|')[0].strip() for x in data]
    speaker = [x.split('|')[2].strip() for x in data]
    speaker_dict = set()

    speaker_dict = hp.data.speakers

    n = len(speaker_dict)
    print(speaker_dict)
    speaker_to_idx = {spk: idx for idx, spk in enumerate(speaker_dict)}

    squares = [0. for i in range(n)]
    means = [0. for i in range(n)]
    frame_count = [0 for i in range(n)]

    for i, fpath in enumerate(tqdm.tqdm(wavdir)):

        spk_idx = speaker_to_idx[speaker[i]]
        square, sum, length = _process_speaker(os.path.join(hp.data.train_dir, fpath),
                                               hp.audio.sampling_rate,
                                               hp.audio.filter_length, hp.audio.hop_length, hp.audio.f0_min,
                                               hp.audio.f0_max, hp.audio.harm_thresh)
        squares[spk_idx] += square
        means[spk_idx] += sum
        frame_count[spk_idx] += length

    result = []
    for i in range(n):
        u = []
        u.append(speaker_dict[i])
        if frame_count[i] == 0:
            avg = 0.0
            avg_sq = 0.0
        else:
            avg = means[i] / frame_count[i]
            avg_sq =squares[i] / frame_count[i]
        u.append(avg)
        u.append(math.sqrt(avg_sq - avg**2))
        result.append(u)

    write_metadata(result, args.output_filename)
