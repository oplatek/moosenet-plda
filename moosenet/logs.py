"""
Pytorch Lightning support multiple dataloaders for validation

We use the following dataloaders:
    - wholeset dataloader to compute statistics on whole validation set
    - valrnd10 dataloader to store samples to wandb to debug in detail few examples
"""
from collections import defaultdict
import logging
import torch
import wandb
from lhotse.utils import ifnone
from lhotse import CutSet
from typing import Dict, Optional
import pytorch_lightning as pl
import torchaudio.functional as F
import torchaudio.transforms as T
import matplotlib.pyplot as plt
from torchmetrics.functional import (
    mean_squared_error,
    spearman_corrcoef,
    pearson_corrcoef,
)

from moosenet.collate import CollateMOS, CollateFbank, CollatePitch
from moosenet.plot import plot_pitch, plot_kaldi_pitch, spec_to_img


def training_step_end(self: pl.LightningModule, outputs):
    d = outputs
    B = d[
        "num_cuts"
    ]  # Be sure to apply only where no cut out of batch masking takes place

    moss = d["stats"].pop("moss", None)
    if moss is not None:
        self.log(
            "train/MOSS",
            moss,
            logger=True,
            prog_bar=True,
            batch_size=B,
            on_step=False,
            on_epoch=True,
        )
    # Try to extract decoder final_mos
    final_mos = d["stats"].pop("dec_mos_final", None)
    # check if encoder is not predicting final_mos
    final_mos = d["stats"].pop("final_mos", None) if final_mos is None else final_mos
    if final_mos is not None:
        self.log(
            "train/MOS",
            final_mos,
            logger=True,
            prog_bar=True,
            batch_size=B,
            on_step=False,
            on_epoch=True,
        )

    for k, v in d["stats"].items():
        # TODO batch_size is incorrect for some metrics here - IGNORING now
        # some metrics cannot be computed on all cuts in batch -> batch_size is smaller
        self.log(f"train/{k}", v, logger=True, prog_bar=False, batch_size=B)

    return outputs


def validation_step_end(self: pl.LightningModule, outputs: Dict):
    d = outputs
    # set batch size according to e.g. this tensors

    # Introduce ALL_loss prefix
    # the current loss start using for clean and remember to distrust any experiments prior 114

    B = d[f"true_{CollateMOS.MOS_FINAL}"].shape[0]
    num_clean = d["num_clean_cuts"]

    # Try to extract decoder final_mos
    final_mos = d["stats"].pop("dec_mos_final", None)
    # check if encoder is not predicting final_mos
    final_mos = d["stats"].pop("final_mos", None) if final_mos is None else final_mos
    if final_mos is not None:
        self.log(
            "val/MOS",
            final_mos,
            logger=True,
            prog_bar=True,
            batch_size=B,
            on_step=False,
            on_epoch=True,
        )
    for k, v in d["stats"].items():
        self.log(f"val/{k}", v, logger=True, prog_bar=False, batch_size=B)

    # evaluate only on clean data with GT MOSS
    mos_final = d[f"pred_{CollateMOS.MOS_FINAL}"]
    true_mos_final = d[f"true_{CollateMOS.MOS_FINAL}"]
    assert mos_final.shape[0] == num_clean, (num_clean, mos_final.shape[0])
    assert true_mos_final.shape[0] == num_clean, (num_clean, true_mos_final.shape[0])
    mos_final = mos_final.float()  # converting always to float32
    self.val_final_mos_mse(mos_final, true_mos_final)
    self.val_final_mos_spearman(mos_final, true_mos_final)
    self.val_final_mos_pearson(mos_final, true_mos_final)

    # https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.log # noqa
    # Default behaviour on_step=False, on_epoch=True for validation
    self.log("val/final_mos_mse", self.val_final_mos_mse, batch_size=num_clean)
    self.log("val/final_mos_pcc", self.val_final_mos_pearson, batch_size=num_clean)
    self.log("val/final_mos_scc", self.val_final_mos_spearman, batch_size=num_clean)


def step_vizualize(logwandb, log_table, outputs: Dict, prefix: Optional[str] = None):
    # Explore https://docs.wandb.ai/guides/track/log/plots#matplotlib-and-plotly-plots
    # https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W%26B_Dataset_Visualization.ipynb#scrollTo=VlH4Et6xOIAo
    d = outputs

    prefix = ifnone(prefix, "viz")
    # TODO add MOS score final to the name
    # TODO plot predicted pitch

    cuts: CutSet = d["cuts"]
    num_clean = d["num_clean_cuts"]
    clean_cuts = cuts.subset(first=num_clean)
    pred_mos_final = d[f"pred_{CollateMOS.MOS_FINAL}"].float()  # convert to float32
    mos_final = d[f"true_{CollateMOS.MOS_FINAL}"]

    predmix_mos_final = d[
        f"predmix_{CollateMOS.MOS_FINAL}"
    ].float()  # convert to float32
    predmix_mos_final_var = d[
        f"predmix_{CollateMOS.MOS_VAR_FINAL}"
    ].float()  # convert to float32

    data = []
    for i, c in enumerate(cuts):
        C = "C" if i < num_clean else "D"  # C for clean D for dirty
        data.append(
            (
                f"{C}_{c.id}",
                predmix_mos_final[i].item(),
                predmix_mos_final_var[i].item(),
            )
        )

    log_table(
        f"{prefix}/predictions/", columns=["cid", "PredMOS", "PredVar"], data=data
    )

    # TODO check it still works
    m_frame_moss = d.get(f"pred_{CollateMOS.FRAMES_MOScores}", None)
    if m_frame_moss is not None:
        m_true_frame_moss = d[f"true_{CollateMOS.FRAMES_MOScores}"]
        m_frame_moss = m_frame_moss.float()  # converting always to float32
        frame_moss_lens = d[CollateMOS.FRAMES_MOScores_lens]

    data = []
    for i, c in enumerate(clean_cuts):
        if c.duration < 0.2:  # Skip short cuts. How much is one encoder step?
            continue
        # Number per cut
        final_mos_mse = mean_squared_error(pred_mos_final[i], mos_final[i]).item()
        # TODO figure out over what to collect stats if I am interested in individual utt: frames?
        # final_mos_spearman = spearman_corrcoef(pred_mos_final[i], mos_final[i])
        # final_mos_pearson = pearson_corrcoef(pred_mos_final[i], mos_final[i])
        if m_frame_moss is not None:
            # Sequence of numbers per cut
            moss_mse = mean_squared_error(m_frame_moss[i], m_true_frame_moss[i]).item()
        else:
            moss_mse = -6

        data.append((c.id, moss_mse, final_mos_mse))

        if m_frame_moss is not None:
            plt.plot(m_frame_moss[i][: frame_moss_lens[i]].cpu())
            plt.ylabel("Frame_MOSS")
            logwandb.log({f"{prefix}/PredictedFrameMOSS/C_{c.id}": plt})
            plt.close("all")

    log_table(
        f"{prefix}/metrics",
        columns=["cid", "FrameMOSMSE", "FinalMOSMSE"],
        data=data,
    )


def vizualize_ground_truth(logwandb, outputs: Dict):
    # Explore https://docs.wandb.ai/guides/track/log/plots#matplotlib-and-plotly-plots
    # https://colab.research.google.com/github/wandb/examples/blob/master/colabs/datasets-predictions/W%26B_Dataset_Visualization.ipynb#scrollTo=VlH4Et6xOIAo

    d = outputs  # reduce reading/typing overhead :-P

    cuts = d["cuts"]
    num_clean = d["num_clean_cuts"]

    fbanks, fbanks_lens = (
        d[CollateFbank.FBANK_FEATURES],
        d[CollateFbank.FBANK_FEATURES_LENS],
    )
    if fbanks is None:
        # Fbanks are not needed losses or the model input
        # Lets compute them just for vizualizing spectrogram
        fbank_coll = CollateFbank()
        fbank_d = fbank_coll(d["audios"], cuts)
        fbanks, fbanks_lens = (
            fbank_d[CollateFbank.FBANK_FEATURES],
            fbank_d[CollateFbank.FBANK_FEATURES_LENS],
        )

    spectrograms = spec_to_img(fbanks, fbanks_lens)
    audios = d["audios"]  # expect dataset with include_audios set
    pitch = d.get(CollatePitch.PITCHES, None)
    pitch_lens = d.get("fixed_pitch_lens", None)

    Bmos_scores = [d.values() for d in CollateMOS.MOS(d)]
    for i, (c, spec, mos_scores) in enumerate(zip(cuts, spectrograms, Bmos_scores)):
        is_clean = i < num_clean
        C = "C" if is_clean else "D"  # C for clean D for dirty
        name = f"{C}_{c.id}_MSEtrue_{'-'.join([str(s) for s in mos_scores])}"
        plt.matshow(spec)
        logwandb.log({f"gt/spec/{name}": plt})
        plt.close("all")

        sample_rate = c.sampling_rate
        audio = audios[i].cpu()

        if pitch is not None:
            pitch_feature = pitch[i][: pitch_lens[i]].cpu()
            ptch, nfcc = pitch_feature[..., 0], pitch_feature[..., 1]
            plot_kaldi_pitch(plt, audio, sample_rate, ptch, nfcc)
            logwandb.log({f"gt/pitch/{name}": plt})
        plt.close("all")

        # TODO plot MCD if we have reference audio
        # TODO does voicemoc_main1 do not have ANY reference audios?

        logwandb.log(
            {f"gt/audio/{name}": wandb.Audio(audio.squeeze(), sample_rate=sample_rate)}
        )
