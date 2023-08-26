import click
from tqdm import tqdm
from itertools import combinations
import logging
import numpy as np
import os
import csv
from typing import Tuple, Optional
import wandb
from pathlib import Path
from lhotse.utils import ifnone
from moosenet.results import process_results, system_scores
from moosenet.collate import CollateMOS
from moosenet.bin import moosenet
from collections import defaultdict


def read_file(filepath):
    with open(filepath, "r", newline="") as csvfile:
        rows = list(csv.reader(csvfile, delimiter=","))
    return dict(sorted((row[0], float(row[1])) for row in rows))


@moosenet.command()
@click.argument("name")
@click.argument("dataset_name")
@click.argument("gold")
@click.argument("results", nargs=-1, type=str)
@click.option("--outdir", type=str)
@click.option("-k", "--k", type=int, default=-1)
def ensemble_results(
    name: str,
    dataset_name: str,
    gold: str,
    results: Tuple[str],
    outdir: Optional[str],
    k: int = -1,
):
    assert dataset_name in [
        "voicemos_main1_test",
        "voicemos_main1_dev",
        "voicemos_main1_train",
        "voicemos_ood1_labeledonly_test",
    ], dataset_name

    k = len(results) if k == -1 else k
    assert 0 < k <= len(results), f"{k=} vs {len(results)=}"

    outdir = Path(ifnone(outdir, "./results/ckpts")).resolve()
    logging.info(f"Preparing ensemble from {len(results)} hypotheses")
    logging.debug(f"{results=}")
    with open(outdir / "data.txt", "wt") as w:
        w.write(f"{gold=}\n{results=}\n")

    goldd = read_file(gold)
    wav_names = list(goldd.keys())
    # we stripped the wav suffix in lhotse preprocessing
    wav_names = [w[:-4] if w.endswith(".wav") else w for w in wav_names]
    sys_names = [CollateMOS.system_name(wn) for wn in wav_names]
    true_mean_scores = list(goldd.values())
    true_sys_mean_scores = defaultdict(list)
    for sn, score in zip(sys_names, true_mean_scores):
        true_sys_mean_scores[sn].append(score)

    raw_results = [read_file(p) for p in results]
    hypotheses = defaultdict(list)
    for r, rpath in zip(raw_results, results):
        hyp_names = list(r.keys())
        assert (
            wav_names == hyp_names
        ), f"utt ids missing for {rpath} and {gold=}\n{set(wav_names) - set(hyp_names)}"
        for wavid, mean_score in r.items():
            hypotheses[wavid].append(mean_score)

    idx_combinations = combinations(list(range(len(results))), k)
    for i, idxs in tqdm(enumerate(idx_combinations)):
        predict_mean_scores = [
            np.mean(np.array(scores)[list(idxs)])
            for wavid, scores in hypotheses.items()
        ]
        predict_sys_mean_scores = system_scores(wav_names, predict_mean_scores)

        wandb_logger = wandb.init(
            name=f"ENSEMBLE_{name}_{i+1}_{k}",
            entity="moosenet",
            project="voicemos_inf",
            dir=str(outdir),
            reinit=True,
        )

        process_results(
            wav_names,
            predict_mean_scores,
            true_mean_scores,
            predict_sys_mean_scores,
            true_sys_mean_scores,
            outdir,
            name,
            dataset_name,
            wandb_logger,
        )
        wandb_logger.finish()
