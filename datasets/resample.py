import os
import glob
import tqdm
from itertools import repeat
from multiprocessing import Pool, freeze_support
from argparse import ArgumentParser

def resampling(wavdir, sr):
    newdir = wavdir.replace('.wav', '-22k.wav')
    os.system('ffmpeg -hide_banner -loglevel panic -y -i %s -ar %d %s' % (wavdir, sr, newdir))
    os.system('rm %s'% wavdir)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--sampling_rate', type=int, default=22050,
                        help="target sampling rate to resample")
    parser.add_argument('--num_workers', type=int, default=32,
                        help="number of workers")
    args = parser.parse_args()

    freeze_support()

    input_paths = glob.glob(os.path.join('datasets', '**', '*.wav'), recursive=True)
    with Pool(processes=args.num_workers) as p:
        r = list(tqdm.tqdm(p.starmap(resampling, zip(input_paths, repeat(args.sampling_rate))), total=len(input_paths)))
