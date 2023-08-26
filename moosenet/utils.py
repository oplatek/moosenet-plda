import logging
import glob
from subprocess import DEVNULL, PIPE, run
import torch
import torch.nn.functional as F
import os
from pytorch_lightning import Callback
from lhotse import load_manifest_lazy
from lhotse.utils import Pathlike
from pathlib import Path


class SetSamplerEpoch(Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        logging.info(
            f"Setting new epoch {trainer.current_epoch} to train_sampler in {os.getpid()}"
        )
        trainer.train_dataloader.sampler.set_epoch(trainer.current_epoch)


def length_to_mask(length, max_len=None, dtype=None):
    """length: B.
    return B x max_len.
    If max_len is None, then max of length will be used.

    Credits:
    [1] https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397/2
    """

    assert len(length.shape) == 1, "Length shape should be 1 dimensional."
    max_len = max_len or length.max().item()
    mask = torch.arange(max_len, device=length.device, dtype=length.dtype).expand(
        len(length), max_len
    ) < length.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
    return mask


def mask_encoder_outputs(
    batch, frame_moss, pitch, encoder_steps, device, skip_check=True
):
    """Frame is here actuall encoder_step
    Our ground truth data should match encoder steps

    skip_check: dimensions checks should be applied only for datasets which has all data collated
    and if you want to really check it.
    """

    from moosenet.collate import CollateMOS, CollatePitch

    PITCHES_LENS = CollatePitch.PITCHES_LENS
    PITCHES = CollatePitch.PITCHES

    true_pitch_lens = batch[PITCHES_LENS]
    assert skip_check or torch.all(
        encoder_steps == true_pitch_lens
    ), f"Collation lens differs {true_pitch_lens} vs {encoder_steps}"

    # DO something smarter that use the same MOS score distributed across all the frames
    true_frame_moss = batch[CollateMOS.MOS_FINAL].repeat(1, encoder_steps)
    true_pitch = batch[PITCHES]
    m = length_to_mask(encoder_steps, max_len=frame_moss.shape[1])
    # TODO change to smarter padding value - See also collate classes

    frame_moss[~m] = 0.0
    true_frame_moss[~m] = 0.0

    # TODO there are slight mismatches in number of frames between pitch and fbank features
    #  nice moosenet train --gpus 0 -w 0 --fast_dev_run -e 0 --prefetch_factor 2 --max_batch_duration 2 --data_setup voicemos_main1 -m ConformerFrameProjection # noqa
    # IndexError: The shape of the mask [1, 39, 2] at index 1 does not match the shape of the indexed tensor [1, 38, 2] at index 1  # noqa
    # # B, T -> B, T, 2
    # m2 = m.reshape(m.shape[0], m.shape[1], 1).expand(m.shape[0], m.shape[1], 2)
    # m2 = length_to_mask(true_pitch_lens, max_len=pitch.shape[1])
    # Interpolate does not work easily better to fix collation anyways
    # m2 = F.interpolate(m2, size=pitch.shape)

    min_pitch_T = min(true_pitch.shape[1], pitch.shape[1])
    pitch = pitch[:, :min_pitch_T, :]
    true_pitch = true_pitch[:, :min_pitch_T, :]
    min_maxT = (min_pitch_T * torch.ones(true_pitch_lens.size())).to(device)
    true_pitch_lens = torch.minimum(true_pitch_lens, min_maxT).long()
    m2 = length_to_mask(true_pitch_lens, max_len=min_pitch_T)
    pitch[~m2] = 0.0
    true_pitch[~m2] = 0.0

    return frame_moss, true_frame_moss, pitch, true_pitch, true_pitch_lens


def parse_ckpt_path(ckpt_path):
    ckpt_path = Path(ckpt_path)
    assert ckpt_path.exists(), str(ckpt_path)
    *args, run_name, _project, wandb_hash, checkpoints, filename = ckpt_path.parts
    assert checkpoints == "checkpoints", str(ckpt_path)
    assert len(wandb_hash) == 8, str(ckpt_path)  # 5s22hce9 example of wandb hash
    return run_name, wandb_hash, ckpt_path.stem


def srcc_score_from_path(ckpt_path: Path):
    ckpt_path = Path(ckpt_path)
    assert ckpt_path.exists(), ckpt_path
    ckpt_path = str(ckpt_path)
    # NOTICE scc instead of SRCC
    # 'exp/1019_23-01-10-12-27-08/voicemos_main/1lwbs78q/checkpoints/E06-val_final_mos_final_mos_scc_0.8760.ckpt'
    srcc = float(ckpt_path.rstrip(".ckpt").split("_")[-1])
    return srcc


def get_best_ckpts(d: Path, best_n: int):
    assert best_n > 0, best_n
    ckpts_paths = glob.glob(f"{d}/*_mos_scc_*.ckpt")
    # SRCC the higher the better
    score_paths = list(
        reversed(sorted([(srcc_score_from_path(p), p) for p in ckpts_paths]))
    )[:best_n]
    return [p for s, p in score_paths]


def get_git_commit():
    git_commit = (
        run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            stdout=PIPE,
            stderr=DEVNULL,
        )
        .stdout.decode()
        .rstrip("\n")
        .strip()
    )
    return git_commit


def is_dirty_git_commit():
    dirty_commit = (
        len(
            run(
                ["git", "diff", "--shortstat"],
                check=True,
                stdout=PIPE,
                stderr=DEVNULL,
            )
            .stdout.decode()
            .rstrip("\n")
            .strip()
        )
        > 0
    )
    return dirty_commit


def get_MOSModel_cls(model_name: str):
    """Helper function useful for experimenting""" 
    from moosenet.models import SSL

    if model_name == "SSL":
        Model = SSL
    else:
        raise ValueError(f"Unsupported model class {model_name}")
    return Model
