import os
import tqdm
import torch
from torch.utils.data import DataLoader
import shutil
import argparse
import pytorch_lightning as pl
from omegaconf import OmegaConf

from synthesizer import Synthesizer
from datasets.text import Language
from datasets import TextMelDataset, text_mel_collate


META_DIR = 'gta_metadata'

class GtaExtractor(object):
    def __init__(self, args):
        self.args = args
        self._load_checkpoint(args.checkpoint_path, args.config)
        self.trainloader = self._gen_dataloader(
            self.hp.data.train_dir, self.hp.data.train_meta)
        self.valloader = self._gen_dataloader(
            self.hp.data.val_dir, self.hp.data.val_meta)

    def _gen_hparams(self, config_paths):
        # generate hparams object for pl.LightningModule
        parser = argparse.ArgumentParser()
        parser.add_argument('--config')
        args = parser.parse_args(['--config', config_paths])
        return args

    def _load_checkpoint(self, checkpoint_path, model_config_path):
        args_temp = self._gen_hparams(model_config_path)
        self.model = Synthesizer(args_temp).cuda()
        self.hp = self.model.hp
        self.lang = Language(self.hp.data.lang, self.hp.data.text_cleaners)

        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        self.model.freeze()
        del checkpoint
        torch.cuda.empty_cache()

    def _gen_dataloader(self, data_dir, data_meta):
        dataset = TextMelDataset(
            self.hp, data_dir, data_meta, train=False, norm=False, use_f0s = True)

        return DataLoader(dataset, batch_size=self.hp.train.batch_size, shuffle=False,
                          num_workers=self.hp.train.num_workers,
                          collate_fn=text_mel_collate, pin_memory=False, drop_last=False)

    def main(self):
        self.extract_and_write_meta('val')
        self.extract_and_write_meta('train')

    def extract_and_write_meta(self, mode):
        assert mode in ['train', 'val']

        dataloader = self.trainloader if mode == 'train' else self.valloader
        desc = 'Extracting GTA mel of %s data' % mode
        meta_list = list()
        for batch in tqdm.tqdm(dataloader, desc=desc):
            temp_meta = self.extract_gta_mels(batch, mode)
            meta_list.extend(temp_meta)

        meta_path = self.hp.data.train_meta if mode == 'train' else self.hp.data.val_meta
        meta_filename = os.path.basename(meta_path)
        new_meta_filename = 'gta_' + meta_filename
        new_meta_path = os.path.join('datasets', META_DIR, new_meta_filename)

        os.makedirs(os.path.join('datasets', META_DIR), exist_ok=True)
        with open(new_meta_path, 'w', encoding='utf-8') as f:
            for wavpath, speaker in meta_list:
                f.write('%s||%s\n' % (wavpath, speaker))

        print('Wrote %d of %d files to %s' % \
            (len(meta_list), len(dataloader.dataset), new_meta_path))

    @torch.no_grad()
    def extract_gta_mels(self, batch, mode):
        text, mel_source, speakers, f0_padded, input_lengths, output_lengths, max_input_len, savepaths = batch
        text = text.cuda()
        mel_source = mel_source.cuda()
        speakers = speakers.cuda()
        f0_padded = f0_padded.cuda()
        input_lengths = input_lengths.cuda()
        output_lengths = output_lengths.cuda()
        max_input_len = max_input_len.cuda()

        ling_s, alignment = self.model.forward(text, mel_source, input_lengths, output_lengths, max_input_len)
        mask = self.model.get_cnn_mask(output_lengths)
        residual = self.model.f0_encoder(f0_padded)
        ling_s = torch.cat((ling_s, residual), dim=1)  # [B, chn.encoder+chn.residual_out, T]
        z_s = self.model.speaker(mel_source, output_lengths)
        mel_s_s = self.model.decoder(ling_s, z_s, mask)

        return self.store_mels_in_savepaths(
            mel_s_s, alignment, input_lengths, output_lengths, savepaths, speakers, mode)

    def store_mels_in_savepaths(self,
        mel_postnet, alignment, input_lengths, output_lengths, savepaths, speakers, mode):
        mels = mel_postnet.detach().cpu()
        alignment = alignment.detach().cpu()
        input_lengths = input_lengths.cpu()
        output_lengths = output_lengths.cpu()
        speakers = speakers.cpu().tolist()

        temp_meta = list()
        for i, path in enumerate(savepaths):
            attention = alignment[i]
            t_enc = input_lengths[i]
            t_dec = output_lengths[i]
            speaker_id = speakers[i]
            speaker = self.hp.data.speakers[speaker_id]

            mel = mels[i][:, :t_dec].clone()

            torch.save(mel, path)
            if mel.size(1) < self.args.min_mel_length:
                continue

            # so now, mel is sufficiently long, and alignment looks good.
            # let's write the mel path to metadata.
            root_dir = self.hp.data.train_dir if mode == 'train' \
                else self.hp.data.val_dir
            rel_path = os.path.relpath(path, start=root_dir)
            wav_path = rel_path.replace('.gta', '')
            temp_meta.append((wav_path, speaker))

        return temp_meta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', nargs=2, type=str, required=True,
                        help="path of configuration yaml file")
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help="path of checkpoint to use for extracting GTA mel")
    parser.add_argument('-m', '--min_mel_length', type=int, default=33,
                        help="minimal length of mel spectrogram. (segment_length // hop_length + 1) expected.")
    args = parser.parse_args()

    extractor = GtaExtractor(args)
    extractor.main()
