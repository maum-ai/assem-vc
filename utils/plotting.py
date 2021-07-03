import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def plot_alignments(alignment1, alignment2, transpose=True):
    if transpose:
        alignment1 = alignment1.T
        alignment2 = alignment2.T

    fig = plt.figure(figsize=(12, 7))
    
    plt.subplot(211)
    plt.imshow(alignment1, aspect='auto', origin='lower', interpolation='none',
        norm=Normalize(vmin=0.0, vmax=1.0))
    plt.ylabel('Encoder timestep')
    
    plt.subplot(212)
    plt.imshow(alignment2, aspect='auto', origin='lower', interpolation='none',
        norm=Normalize(vmin=0.0, vmax=1.0))
    plt.xlabel('Decoder timestep')
    plt.ylabel('Encoder timestep')

    plt.subplots_adjust(bottom=0.1, right=0.88, top=0.9)
    cax = plt.axes([0.9, 0.1, 0.02, 0.8])
    plt.colorbar(cax=cax)
    return fig

def plot_spectrograms(mel_pred, mel_postnet, mel_target):
    fig = plt.figure(figsize=(12, 9))
    
    plt.subplot(311)
    plt.imshow(mel_pred, aspect='auto', origin='lower', interpolation='none',
        norm=Normalize(vmin=-5.0, vmax=1.0))
    plt.ylabel('before postnet')

    plt.subplot(312)
    plt.imshow(mel_postnet, aspect='auto', origin='lower', interpolation='none',
        norm=Normalize(vmin=-5.0, vmax=1.0))
    plt.ylabel('after postnet')

    plt.subplot(313)
    plt.imshow(mel_target, aspect='auto', origin='lower', interpolation='none',
        norm=Normalize(vmin=-5.0, vmax=1.0))
    plt.ylabel('target mel')
    plt.xlabel('time frames')
    
    plt.subplots_adjust(bottom=0.1, right=0.88, top=0.9)
    cax = plt.axes([0.9, 0.1, 0.02, 0.8])
    plt.colorbar(cax=cax)
    return fig

# def plot_gate(gate_target, gate_predicted):
#     fig = plt.figure(figsize=(12, 3))
#     plt.scatter(range(len(gate_target)), gate_target, alpha=0.5,
#             color='green', marker='.', s=1, label='target')
#     plt.scatter(range(len(gate_predicted)), gate_predicted, alpha=0.5,
#             color='red', marker='.', s=1, label='predicted')
#     plt.xlabel('Frames')
#     plt.ylabel('Gate value')
#     plt.xlim(0, len(gate_target))
#     plt.legend(loc=0)
#     plt.subplots_adjust(bottom=0.1, right=0.88, top=0.9)
#     return fig

def plot_conversions(mel_source, mel_s_s, mel_s_t):
    fig = plt.figure(figsize=(12, 9))

    plt.subplot(311)
    plt.imshow(mel_source, aspect='auto', origin='lower', interpolation='none',
        norm=Normalize(vmin=-5.0, vmax=1.0))
    plt.ylabel('mel_source')

    plt.subplot(312)
    plt.imshow(mel_s_s, aspect='auto', origin='lower', interpolation='none',
        norm=Normalize(vmin=-5.0, vmax=1.0))
    plt.ylabel('mel_s_s')

    plt.subplot(313)
    plt.imshow(mel_s_t, aspect='auto', origin='lower', interpolation='none',
        norm=Normalize(vmin=-5.0, vmax=1.0))
    plt.ylabel('mel_s_t')
    plt.xlabel('time frames')
    
    plt.subplots_adjust(bottom=0.1, right=0.88, top=0.9)
    cax = plt.axes([0.9, 0.1, 0.02, 0.8])
    plt.colorbar(cax=cax)
    return fig

def plot_residual(residual):
    fig = plt.figure(figsize=(12, 3))
    for feat in residual:
        plt.plot(range(len(feat)), feat)
    plt.xlim(0, len(residual[0]))
    plt.xlabel('time frames')
    plt.ylabel('residual info')
    plt.subplots_adjust(bottom=0.1, right=0.88, top=0.9)
    return fig
