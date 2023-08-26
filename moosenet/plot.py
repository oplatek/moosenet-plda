import torch
import torchaudio.functional as F
import numpy as np

import matplotlib.pyplot as pyplot


def plot_pitch(plt, waveform, sample_rate, pitch):
    # https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html
    figure, axis = plt.subplots(1, 1)
    axis.set_title("Pitch Feature")
    axis.grid(True)

    end_time = waveform.shape[1] / sample_rate
    time_axis = torch.linspace(0, end_time, waveform.shape[1])
    axis.plot(time_axis, waveform[0], linewidth=1, color="gray", alpha=0.3)

    axis2 = axis.twinx()
    time_axis = torch.linspace(0, end_time, pitch.shape[1])
    axis2.plot(time_axis, pitch[0], linewidth=2, label="Pitch", color="green")

    axis2.legend(loc=0)


def plot_kaldi_pitch(plt, waveform, sample_rate, pitch, nfcc):
    figure, axis = plt.subplots(1, 1)
    axis.set_title("Kaldi Pitch Feature")
    axis.grid(True)
    assert len(waveform.shape) == 1, str(waveform.shape)
    assert len(pitch.shape) == 1, str(pitch.shape)
    assert len(nfcc.shape) == 1, str(nfcc.shape)

    end_time = waveform.shape[0] / sample_rate
    time_axis = torch.linspace(0, end_time, waveform.shape[0])
    axis.plot(time_axis, waveform, linewidth=1, color="gray", alpha=0.3)

    time_axis = torch.linspace(0, end_time, pitch.shape[0])
    ln1 = axis.plot(time_axis, pitch, linewidth=2, label="Pitch", color="green")
    axis.set_ylim((-1.3, 1.3))

    axis2 = axis.twinx()
    time_axis = torch.linspace(0, end_time, nfcc.shape[0])
    ln2 = axis2.plot(
        time_axis, nfcc, linewidth=2, label="NFCC", color="blue", linestyle="--"
    )

    lns = ln1 + ln2
    labels = [ln.get_label() for ln in lns]
    axis.legend(lns, labels, loc=0)


def spec_to_img(mels: torch.Tensor, lens: torch.Tensor, trim=True, matshow=False):
    """
    :param mels: (B, T, n_melbins)
    :param lens: (B,)
    """
    mels = mels.cpu().float()
    # B, T, n_mel  -> B, n_mel, T
    mels = mels.transpose(1, 2)
    mels = np.flip(mels.numpy(), 1)
    lens = lens.cpu().numpy()
    for i, d in enumerate(lens):
        img = mels[i, :, :d] if trim else mels[i, :, :]
        if matshow:
            img = pyplot.matshow(img)
            pyplot.close("all")
        yield img
