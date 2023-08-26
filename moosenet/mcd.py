#!/usr/bin/env python
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
# Adapted from espnet/utils/mcd_calculate.py

import numpy as np
import pysptk
import pyworld as pw
import scipy
from fastdtw import fastdtw
from scipy.io import wavfile
from scipy.signal import firwin, lfilter
from lhotse.utils import ifnone


def low_cut_filter(x, fs, cutoff=70):
    """FUNCTION TO APPLY LOW CUT FILTER

    Args:
        x (ndarray): Waveform sequence
        fs (int): Sampling frequency
        cutoff (float): Cutoff frequency of low cut filter

    Return:
        (ndarray): Low cut filtered waveform sequence
    """

    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist

    # low cut filter
    fil = firwin(255, norm_cutoff, pass_zero=False)
    lcf_x = lfilter(fil, 1, x)

    return lcf_x


def spc2npow(spectrogram):
    """Calculate normalized power sequence from spectrogram

    Parameters
    ----------
    spectrogram : array, shape (T, `fftlen / 2 + 1`)
        Array of spectrum envelope

    Return
    ------
    npow : array, shape (`T`, `1`)
        Normalized power sequence

    """

    # frame based processing
    npow = np.apply_along_axis(_spvec2pow, 1, spectrogram)

    meanpow = np.mean(npow)
    npow = 10.0 * np.log10(npow / meanpow)

    return npow


def _spvec2pow(specvec):
    """Convert a spectrum envelope into a power

    Parameters
    ----------
    specvec : vector, shape (`fftlen / 2 + 1`)
        Vector of specturm envelope |H(w)|^2

    Return
    ------
    power : scala,
        Power of a frame

    """

    # set FFT length
    fftl2 = len(specvec) - 1
    fftl = fftl2 * 2

    # specvec is not amplitude spectral |H(w)| but power spectral |H(w)|^2
    power = specvec[0] + specvec[fftl2]
    for k in range(1, fftl2):
        power += 2.0 * specvec[k]
    power /= fftl

    return power


def extfrm(data, npow, power_threshold=-20):
    """Extract frame over the power threshold

    Parameters
    ----------
    data: array, shape (`T`, `dim`)
        Array of input data
    npow : array, shape (`T`)
        Vector of normalized power sequence.
    power_threshold : float, optional
        Value of power threshold [dB]
        Default set to -20

    Returns
    -------
    data: array, shape (`T_ext`, `dim`)
        Remaining data after extracting frame
        `T_ext` <= `T`

    """

    T = data.shape[0]
    if T != len(npow):
        raise ("Length of two vectors is different.")

    valid_index = np.where(npow > power_threshold)
    extdata = data[valid_index]
    assert extdata.shape[0] <= T

    return extdata


def world_extract(
    x,
    sample_rate,
    f0min=40,
    f0max=800,
    shift_ms=5,
    mcep_dim=41,
    mcep_alpha=0.41,
    fftl=1024,
):
    """Feature extraction using world vocoder features"""
    # fs, x = wavfile.read(wav_path)
    fs = sample_rate
    x = np.array(x, dtype=np.float64)
    x = low_cut_filter(x, fs)

    # extract features
    f0, time_axis = pw.harvest(
        x, fs, f0_floor=f0min, f0_ceil=f0max, frame_period=shift_ms
    )
    sp = pw.cheaptrick(x, f0, time_axis, fs, fft_size=fftl)
    ap = pw.d4c(x, f0, time_axis, fs, fft_size=fftl)
    mcep = pysptk.sp2mc(sp, mcep_dim, mcep_alpha)
    npow = spc2npow(sp)

    return {
        "sp": sp,
        "mcep": mcep,
        "ap": ap,
        "f0": f0,
        "npow": npow,
    }


def calculate_mcd(x, y, sample_rate, gt_feats=None, cvt_feats=None):
    # extract ground truth and converted features
    gt_feats = ifnone(gt_feats, world_extract(x, sample_rate))
    cvt_feats = ifnone(cvt_feats, world_extract(y, sample_rate))

    # VAD & DTW based on power
    gt_mcep_nonsil_pow = extfrm(gt_feats["mcep"], gt_feats["npow"])
    cvt_mcep_nonsil_pow = extfrm(cvt_feats["mcep"], cvt_feats["npow"])
    _, path = fastdtw(
        cvt_mcep_nonsil_pow,
        gt_mcep_nonsil_pow,
        dist=scipy.spatial.distance.euclidean,
    )
    twf_pow = np.array(path).T

    # MCD using power-based DTW
    cvt_mcep_dtw_pow = cvt_mcep_nonsil_pow[twf_pow[0]]
    gt_mcep_dtw_pow = gt_mcep_nonsil_pow[twf_pow[1]]
    diff2sum = np.sum((cvt_mcep_dtw_pow - gt_mcep_dtw_pow) ** 2, 1)
    mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum), 0)
    return mcd
