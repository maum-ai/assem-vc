import random
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
from utils.plotting import plot_alignments, plot_spectrograms, plot_conversions, plot_residual


class TacotronLogger(TensorBoardLogger):
    def __init__(self, save_dir, name='default', version=None, **kwargs):
        super().__init__(save_dir, name, version, **kwargs)

    @rank_zero_only
    def log_loss(self, loss, mode, step, name=''):
        name = '.loss' + ('' if name=='' else '_'+name)
        self.experiment.add_scalar(mode + name, loss, step)

    @rank_zero_only
    def log_figures(self, mel_pred, mel_postnet, mel_target, alignment, step):
        mel_pred = mel_pred.cpu().detach().numpy()
        mel_postnet = mel_postnet.cpu().detach().numpy()
        mel_target = mel_target.cpu().detach().numpy()
        alignment = alignment.cpu().detach().numpy()

        rand_idx = random.randint(1, len(mel_pred)-1)

        alignment_plot = plot_alignments(alignment[0], alignment[rand_idx], transpose=True)
        self.experiment.add_figure('alignment_fixed_random', alignment_plot, step)

        spectrogram_plot = plot_spectrograms(mel_pred[rand_idx], mel_postnet[rand_idx], mel_target[rand_idx])
        self.experiment.add_figure('mel_spectrograms', spectrogram_plot, step)

    @rank_zero_only
    def log_embedding(self, symbols, embedding, step):
        self.experiment.add_embedding(
            mat=embedding,
            metadata=symbols,
            global_step=step,
            tag='character_embedding')

    @rank_zero_only
    def log_learning_rate(self, learning_rate, step):
        self.experiment.add_scalar('learning_rate', learning_rate, step)


class SynthesizerLogger(TensorBoardLogger):
    def __init__(self, save_dir, name='default', version=None, **kwargs):
        super().__init__(save_dir, name, version, **kwargs)

    @rank_zero_only
    def log_loss(self, loss, mode, step, name=''):
        name = '.loss' + ('' if name=='' else '_'+name)
        self.experiment.add_scalar(mode + name, loss, step)

    @rank_zero_only
    def log_figures(self, mel_source, mel_s_s, mel_s_t, alignment, residual, step):
        mel_source = mel_source.cpu().detach().numpy()
        mel_s_s = mel_s_s.cpu().detach().numpy()
        mel_s_t = mel_s_t.cpu().detach().numpy()
        alignment = alignment.cpu().detach().numpy()
        residual = residual.cpu().detach().numpy()

        rand_idx = random.randint(1, len(mel_source)-1)

        alignment_plot = plot_alignments(alignment[0], alignment[rand_idx], transpose=True)
        self.experiment.add_figure('pretrained_alignment_fixed_random', alignment_plot, step)

        spectrogram_plot = plot_conversions(
            mel_source[rand_idx], mel_s_s[rand_idx], mel_s_t[rand_idx])
        self.experiment.add_figure('conversions', spectrogram_plot, step)

        residual_plot = plot_residual(residual[rand_idx])
        self.experiment.add_figure('residual_info', residual_plot, step)


class DodLogger(TensorBoardLogger):
    def __init__(self, save_dir, name='default', version=None, **kwargs):
        super().__init__(save_dir, name, version, **kwargs)

    @rank_zero_only
    def log_loss(self, loss, mode, step, name=''):
        name = '.loss' + ('' if name=='' else '_'+name)
        self.experiment.add_scalar(mode + name, loss, step)

    @rank_zero_only
    def log_accuracy(self, accuracy, step):
        self.experiment.add_scalar('accuracy', accuracy, step)
