from moosenet.bin import moosenet
from statistics import mean
import numpy as np
from pathlib import Path
from moosenet.collate import CollateMOS
from lhotse.utils import ifnone
from moosenet.datasetup import get_manifest
import click
from moosenet.results import gen_results


@moosenet.command()
@click.argument("data_setup", type=str)  # help="voicemos_main1"
@click.argument("split", type=str)  # help="dev, test"
@click.option("--outdir", type=str, default=None)
def plot_true(data_setup: str, split: str, outdir: str):
    outdir = Path(ifnone(outdir, "./results/ckpts")).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    assert split in ["train", "dev"], split
    train_cuts, val_cuts, _val_egs_manifests, _test_cuts = get_manifest(data_setup)
    cuts = train_cuts if split == "train" else val_cuts
    cuts = [c for c in cuts]
    batch = {"cuts": cuts}

    mos_values = [list(mosd.values()) for mosd in CollateMOS.MOS(batch)]
    wav_names = [c.id for c in cuts]

    optimistic_scores = np.array([max(scores) for scores in mos_values])
    pesimistic_scores = np.array([min(scores) for scores in mos_values])
    avg_true_scores = np.array([mean(scores) for scores in mos_values])

    sys_names = CollateMOS.system_names(batch)

    listeners_names = None  # we can simply ignore

    gen_results(
        [(optimistic_scores, wav_names, listeners_names, sys_names, avg_true_scores)],
        outdir,
        "HUMAN_ANNOTATOR_OPTIMIST",
        f"{data_setup}_{split}",
    )

    gen_results(
        [(pesimistic_scores, wav_names, listeners_names, sys_names, avg_true_scores)],
        outdir,
        "HUMAN_ANNOTATOR_PESIMIST",
        f"{data_setup}_{split}",
    )

    gen_results(
        [(avg_true_scores, wav_names, listeners_names, sys_names, avg_true_scores)],
        outdir,
        "HUMAN_ANNOTATOR_MEAN",
        f"{data_setup}_{split}",
    )
