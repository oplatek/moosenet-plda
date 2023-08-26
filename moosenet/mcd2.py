import logging
from typing import Any, Callable, Optional

import numpy as np
import torch
from fastdtw import fastdtw
from lhotse.utils import ifnone
from torch import Tensor, tensor
from torchaudio.functional import create_dct

# using torchmetrics implementation in our package
from torchmetrics.audio import SI_SDR, SI_SNR  # noqa
from torchmetrics.metric import Metric


class MelCepstralDistortion(Metric):
    """
    MelCepstralDistortion is distance (RMSE) metric for two
    sequences of MFCCs features aligned using DTW

    Limitations:
    -  "... Modified vowels with sharp and narrow 3rd or 4th formant are easily perceptually
    distinguishable from the original vowels... " [1]
    - " ... Fail to  small difference between vowels’ first formants ... by the MFCC based measures ... "[1]
    - " .. spectral differences of the higher frequencies regions were clearly audible in subjective listening test,
        but were poorly measured by the MFCC based measures ... " [1]
    - recommended 220 Mel bins [1] vs 80 used

    Literature:
    1. Perceptual Significance of Cepstral Distortion Measures in Digital Speech Processing;
        Antonio Vasilijević & Davor Petrinović
    2. SYNTHESIZER VOICE QUALITY OF NEW LANGUAGES CALIBRATED WITH MEAN MEL CEPSTRAL DISTORTION,
        Kominek John & Schultz Tanja & Alan Black
    """

    # These are accumulators. See self.add_state. Here for Mypy compliance
    sum_mcd: Tensor
    frames: Tensor

    def __init__(
        self,
        n_fbank: int,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable[[Tensor], Tensor]] = None,
        n_mfcc: Optional[int] = None,
    ) -> None:
        """
        MelCepstralDistortion runs on log Mel Filter banks features.

        n_fbank: Number of fbanks for the log filter bank features
        n_mfcc: Number of MFCC computed from the log Filter bannks influences quality of MCD[1]

        MelCepstralDistortion is based on torchmetrics.Metric
            so two functions should be used:
            - update() to collect inputs on which the metric is computed
            - compute() to output the MCD value for so far collected inputs

        Since torchmetrics.Metric API is designed to support distributed training
        the following parameters are needed:
        - compute_on_step
        - dist_sync_on_step
        - process_group
        - dist_sync_fn
        See https://torchmetrics.readthedocs.io/en/latest/references/modules.html#base-class
        """
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        n_mfcc = ifnone(n_mfcc, n_fbank)
        assert n_mfcc > 1
        assert n_fbank > 1
        self.n_fbank = n_fbank  # Number of log mel filters

        # Number of coefficients used for MFCC
        # In ASR 13 coefficients are typically used,
        # But for MCD is typicaly used 41.
        # but the higher number of MFCCs the better distinction between wovels[1].
        # Since we use 80 Mel fbanks, 80 is the highest reasonable default
        if n_mfcc != 41 and n_mfcc != n_fbank:
            logging.warning(
                f"Value of n_mfcc {n_mfcc} differs from requested specified number of Mel bins"
                "It also differs from 41 which is standard value for MCD. Another standard value"
            )
        self.n_mfcc = n_mfcc

        self.add_state("sum_mcd", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("frames", default=tensor(0), dist_reduce_fx="sum")

        self.dct_mat = create_dct(self.n_mfcc, self.n_fbank, "ortho")

    def update(
        self,
        pred_specs: Tensor,
        target_specs: Tensor,
        pred_lens: Optional[Tensor] = None,
        target_lens: Optional[Tensor] = None,
    ) -> None:
        """
        pred_specs: log Mel fbanks predicted by a VC or TTS system
            shape (T, M) or (B, T, M)
        target_specs: ground truth log Mel fbanks:
            shape (T, M) or (B, T, M)

          where
            T is maximum number of frames in batch
            M is number of log Mel fbanks
            B is batch size

        """

        if pred_specs.dim() == 2:
            pred_specs = pred_specs.unsqueeze(dim=0)
            target_specs = target_specs.unsqueeze(dim=0)
        # Assumes (B, T, M) shape
        assert (
            3 == pred_specs.dim() == target_specs.dim()
        ), f"{pred_specs.shape} vs {target_specs.shape}"
        assert (
            pred_specs.shape[0] == target_specs.shape[0]
        ), f"Batch size mismatch {pred_specs.shape} vs {target_specs.shape}"
        assert (
            pred_specs.shape[2] == target_specs.shape[2]
        ), f"Log mel fbanks num mismatch {pred_specs.shape} vs {target_specs.shape}"

        B = pred_specs.shape[0]

        # maximum lengths
        if pred_lens is None:
            pred_lens = torch.full((B,), pred_specs.shape[1])
        if target_lens is None:
            target_lens = torch.full((B,), target_specs.shape[1])

        dct_mat = self.dct_mat.to(pred_specs.device)

        mcds = torch.empty(B)
        # iterate over individual log Mel spectrograms
        for i in range(B):
            # shape (T, M)
            log_mels_x = pred_specs[i, : pred_lens[i], :]
            log_mels_y = target_specs[i, : target_lens[i], :]

            # Taken from: https://pytorch.org/audio/stable/_modules/torchaudio/transforms.html#MFCC
            #  shape: (.., T, M) -> (.., T, n_mfcc).transpose(..) i.e. (.., n_mfcc, T)
            # See also https://docs.scipy.org/doc/scipy/reference/generated/scipy.fftpack.dct.html
            mfcc_x = torch.matmul(log_mels_x, dct_mat).transpose(-2, -1)
            mfcc_y = torch.matmul(log_mels_y, dct_mat).transpose(-2, -1)

            # Dropping overall recording signal power  - zeroth cepstral dimention.
            # See [1], [2] Section 2.1.
            # shape (T, self.n_mfcc - 1)
            mfcc_x = mfcc_x[1 : self.n_mfcc, :]
            mfcc_y = mfcc_y[1 : self.n_mfcc, :]

            _, path = fastdtw(mfcc_x.cpu(), mfcc_y.cpu(), dist=self._RMSE)
            pathx, pathy = map(list, zip(*path))
            cum_distance = self._RMSE(
                mfcc_x.cpu()[pathx].T, mfcc_y.cpu()[pathy].T
            ).sum()

            # Normalization of the distance per frame is done per dataset in self.compute
            self.sum_mcd += cum_distance
            self.frames += len(pathx)
            mcds[i] = cum_distance / len(pathx)
        return mcds

    @staticmethod
    def _RMSE(s1, s2):
        if torch.is_tensor(s1):
            # The conversion is performed only once
            # The fastdtw returns numpy array
            # for compatible DTW implementation
            # we could use sqrt(torch.nn.functional.mse_loss) instead
            s1, s2 = s1.numpy(), s2.numpy()
        diff = s1 - s2
        # axis 0 - represent MFCC coefficients
        d = np.sqrt(np.sum(diff * diff, axis=0))
        return d

    def compute(self) -> Tensor:
        if self.frames == 0:
            return tensor(float("nan"))
        return self.sum_mcd / self.frames

    @property
    def is_differentiable(self) -> bool:
        return False
