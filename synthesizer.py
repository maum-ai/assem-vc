import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import numpy as np
from omegaconf import OmegaConf

from cotatron import Cotatron
from modules import VCDecoder, SpeakerEncoder, F0_Encoder
from datasets import TextMelDataset, text_mel_collate


class Synthesizer(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters()
        hp_global = OmegaConf.load(hparams.config[0])
        hp_vc = OmegaConf.load(hparams.config[1])
        hp = OmegaConf.merge(hp_global, hp_vc)
        self.hp = hp

        self.num_speakers = len(self.hp.data.speakers)
        self.cotatron = Cotatron(hparams)
        self.f0_encoder = F0_Encoder(hp)
        self.decoder = VCDecoder(hp)
        self.speaker = SpeakerEncoder(hp)

        self.is_val_first = True

    def load_cotatron(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.cotatron.load_state_dict(checkpoint['state_dict'])
        self.cotatron.eval()
        self.cotatron.freeze()

    # this is called after validation/test loop is finished.
    # see the order of hooks being called at :
    # https://pytorch-lightning.readthedocs.io/en/latest/hooks.html
    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)

        self.cotatron.eval()
        self.cotatron.freeze()
        return self

    def forward(self, text, mel_source, input_lengths, output_lengths, max_input_len):
        z_s_cota = self.cotatron.speaker(mel_source, output_lengths)
        text_encoding = self.cotatron.encoder(text, input_lengths)
        z_s_repeated = z_s_cota.unsqueeze(1).expand(-1, text_encoding.size(1), -1)
        decoder_input = torch.cat((text_encoding, z_s_repeated), dim=2)
        _, _, alignment = \
            self.cotatron.decoder(mel_source, decoder_input, input_lengths, output_lengths, max_input_len,
                                  prenet_dropout=0.5, tfrate=False)

        # alignment: [B, T_dec, T_enc]
        # text_encoding: [B, T_enc, chn.encoder]
        linguistic = torch.bmm(alignment, text_encoding)  # [B, T_dec, chn.encoder]
        linguistic = linguistic.transpose(1, 2)  # [B, chn.encoder, T_dec]
        return linguistic, alignment

    def inference(self, text, mel_source, mel_reference, f0_padded):
        device = text.device
        in_len = torch.LongTensor([text.size(1)]).to(device)
        out_len = torch.LongTensor([mel_source.size(2)]).to(device)

        z_s_cota = self.cotatron.speaker.inference(mel_source)
        z_t = self.speaker.inference(mel_reference)

        text_encoding = self.cotatron.encoder.inference(text)
        z_s_repeated = z_s_cota.unsqueeze(1).expand(-1, text_encoding.size(1), -1)
        decoder_input = torch.cat((text_encoding, z_s_repeated), dim=2)
        _, _, alignment = \
            self.cotatron.decoder(mel_source, decoder_input, in_len, out_len, in_len,
                                  prenet_dropout=0.5, no_mask=True, tfrate=False)
        ling_s = torch.bmm(alignment, text_encoding)
        ling_s = ling_s.transpose(1, 2)

        residual = self.f0_encoder(f0_padded)
        ling_s = torch.cat((ling_s, residual), dim=1)

        mel_s_t = self.decoder(ling_s, z_t)
        return mel_s_t, alignment, residual

    def inference_from_z_t(self, text, mel_source, z_t):
        device = text.device
        in_len = torch.LongTensor([text.size(1)]).to(device)
        out_len = torch.LongTensor([mel_source.size(2)]).to(device)

        z_s_cota = self.cotatron.speaker.inference(mel_source)

        text_encoding = self.cotatron.encoder.inference(text)
        z_s_repeated = z_s_cota.unsqueeze(1).expand(-1, text_encoding.size(1), -1)
        decoder_input = torch.cat((text_encoding, z_s_repeated), dim=2)
        _, _, alignment = \
            self.cotatron.decoder(mel_source, decoder_input, in_len, out_len, in_len,
                                  prenet_dropout=0.5, no_mask=True, tfrate=False)
        ling_s = torch.bmm(alignment, text_encoding)
        ling_s = ling_s.transpose(1, 2)

        residual = self.f0_encoder(f0_padded)
        ling_s = torch.cat((ling_s, residual), dim=1)

        mel_s_t = self.decoder(ling_s, z_t)
        return mel_s_t, alignment, residual

    # masking convolution from GAN-TTS (arXiv:1909.11646)
    def get_cnn_mask(self, lengths):
        max_len = torch.max(lengths).item()
        ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
        mask = (ids >= lengths.unsqueeze(1))
        mask = mask.unsqueeze(1)
        return mask  # [B, 1, T], torch.bool

    def training_step(self, batch, batch_idx):
        text, mel_source, speakers, f0_padded, input_lengths, output_lengths, max_input_len, _ = batch

        with torch.no_grad():
            ling_s, _ = self(text, mel_source, input_lengths, output_lengths, max_input_len)

        mask = self.get_cnn_mask(output_lengths)
        residual = self.f0_encoder(f0_padded)
        ling_s = torch.cat((ling_s, residual), dim=1)  # [B, chn.encoder+chn.residual_out, T]
        z_s = self.speaker(mel_source, output_lengths)
        mel_s_s = self.decoder(ling_s, z_s, mask)
        loss_rec = F.mse_loss(mel_s_s, mel_source)
        self.logger.log_loss(loss_rec, mode='train', step=self.global_step, name='rec')

        return {'loss': loss_rec}

    def validation_step(self, batch, batch_idx):
        text, mel_source, speakers, f0_padded, input_lengths, output_lengths, max_input_len, _ = batch

        mask = self.get_cnn_mask(output_lengths)
        ling_s, alignment = self(text, mel_source, input_lengths, output_lengths, max_input_len)

        z_s = self.speaker(mel_source, output_lengths)
        rand_batch_idx = np.random.randint(z_s.size(0), size=z_s.size(0))
        z_t = z_s[rand_batch_idx]

        residual = self.f0_encoder(f0_padded)
        ling_s = torch.cat((ling_s, residual), dim=1)  # [B, chn.encoder+chn.residual_out, T]
        mel_s_s = self.decoder(ling_s, z_s, mask)
        mel_s_t = self.decoder(ling_s, z_t, mask)
        loss_rec = F.mse_loss(mel_s_s, mel_source)

        if self.is_val_first:
            self.is_val_first = False
            self.logger.log_figures(mel_source, mel_s_s, mel_s_t, alignment, f0_padded, self.global_step)

        return {'loss_rec': loss_rec}

    def validation_epoch_end(self, outputs):
        loss_rec = torch.stack([x['loss_rec'] for x in outputs]).mean()

        self.is_val_first = True
        self.log('val_loss', loss_rec)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.decoder.parameters()) \
            + list(self.f0_encoder.parameters()) \
            + list(self.speaker.parameters()),
            lr=self.hp.train.adam.lr,
            weight_decay=self.hp.train.adam.weight_decay,
        )
        return optimizer

    def train_dataloader(self):
        trainset = TextMelDataset(
            self.hp, self.hp.data.train_dir, self.hp.data.train_meta, train=True, norm=False, use_f0s=True)

        return DataLoader(trainset, batch_size=self.hp.train.batch_size, shuffle=True,
                        num_workers=self.hp.train.num_workers,
                        collate_fn=text_mel_collate, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        valset = TextMelDataset(
            self.hp, self.hp.data.val_dir, self.hp.data.val_meta, train=False, norm=False, use_f0s=True)

        return DataLoader(valset, batch_size=self.hp.train.batch_size, shuffle=False,
                        num_workers=self.hp.train.num_workers,
                        collate_fn=text_mel_collate, pin_memory=False, drop_last=False)
