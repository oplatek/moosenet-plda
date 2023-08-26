from collections import defaultdict
from moosenet.plda_utils import MODEL_CONFIDENCE
from subprocess import DEVNULL, PIPE, run
import shutil
from moosenet.utils import is_dirty_git_commit, get_git_commit
from moosenet.collate import CollateMOS
from pathlib import Path
import csv
import wandb
import scipy
import torch
import numpy as np
from typing import List, Dict, Any, Union, Optional

import matplotlib

# Force matplotlib to not use any Xwindows backend.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def upload_predictions(predictions, save_dir, name, dataset_name, wandb_logger):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_prefix = save_dir / f"{dataset_name}_{name}_answer"
    submission_scores = []

    for batch in predictions:
        num_clean_cuts = batch["num_clean_cuts"]
        pred_mean_scores = batch[f"pred_{CollateMOS.MOS_FINAL}"][:num_clean_cuts]
        cuts = [c for i, c in zip(range(num_clean_cuts), batch["cuts"])]
        wav_names = [c.id for c in cuts]
        pred_mean_scores = pred_mean_scores.reshape(pred_mean_scores.shape[0])
        for wav_name, pred_mean_score in zip(wav_names, list(pred_mean_scores)):
            sysid, uttid, _lhotse_part1, _lhotse_part2 = wav_name.split("-")
            wav_name = f"{sysid}-{uttid}"
            submission_scores.append((wav_name, pred_mean_score))

    answer_file = save_prefix.parent / (save_prefix.name + "_answer.txt")
    with open(answer_file, "wt") as f:
        for wavname, score in sorted(submission_scores):
            f.write(f"{wavname},{score}\n")
    wandb_logger.save(str(answer_file))


def system_scores(wav_names, scores):
    sys_scores = defaultdict(list)
    sys_names = [CollateMOS.system_name(wn) for wn in wav_names]
    for sn, score in zip(sys_names, scores):
        sys_scores[sn].append(score)
    return sys_scores


def gen_results(
    predictions,
    save_dir,
    name,
    dataset_name,
    wandb_logger,
):
    """
    Heavily inspired from LDNet inference_for_voicemos.py script
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    predict_mean_scores = []
    true_mean_scores = []
    scores_confidence = []
    wav_names = []

    for batch in predictions:
        num_clean_cuts = batch["num_clean_cuts"]
        pred_mean_scores = batch[f"pred_{CollateMOS.MOS_FINAL}"][:num_clean_cuts]
        cuts = [c for i, c in zip(range(num_clean_cuts), batch["cuts"])]
        btrue_mean_scores = batch[f"true_{CollateMOS.MOS_FINAL}"][:num_clean_cuts]
        # TODO use listeners names
        pred_mean_scores = pred_mean_scores.reshape(pred_mean_scores.shape[0])
        btrue_mean_scores = btrue_mean_scores.reshape(btrue_mean_scores.shape[0])

        predict_mean_scores.extend(pred_mean_scores.tolist())
        true_mean_scores.extend(btrue_mean_scores.tolist())
        for wav_name in [c.id for c in cuts]:
            sysid, uttid, _lhotse_part1, _lhotse_part2 = wav_name.split("-")
            wav_name = f"{sysid}-{uttid}"
            wav_names.append(wav_name)

        if "PLDA" in dataset_name:
            model_confidences = batch[MODEL_CONFIDENCE][:num_clean_cuts]
        else:
            model_confidences = batch[f"pred_{CollateMOS.MOS_VAR_FINAL}"][
                :num_clean_cuts
            ]

        model_confidences = model_confidences.reshape(model_confidences.shape[0])
        for score_confidence in model_confidences.tolist():
            scores_confidence.append(score_confidence)

    predict_sys_mean_scores = system_scores(wav_names, predict_mean_scores)
    true_sys_mean_scores = system_scores(wav_names, true_mean_scores)

    save_prefix = save_dir / f"{dataset_name}_{name}_answer"
    confidence_file = save_prefix.parent / (save_prefix.name + "_model_confidence.txt")
    with open(confidence_file, "wt") as f:
        for wavname, cnfd in sorted(zip(wav_names, scores_confidence)):
            f.write(f"{wavname},{cnfd}\n")

    return process_results(
        wav_names,
        predict_mean_scores,
        true_mean_scores,
        predict_sys_mean_scores,
        true_sys_mean_scores,
        save_dir,
        name,
        dataset_name,
        wandb_logger,
    )


def process_results(
    wav_names,
    predict_mean_scores,
    true_mean_scores,
    predict_sys_mean_scores: Dict[str, List[float]],
    true_sys_mean_scores: Dict[str, List[float]],
    save_dir,
    name,
    dataset_name,
    wandb_logger,
):

    ep_scores = []
    save_prefix = Path(save_dir) / f"{dataset_name}_{name}_answer"
    answer_file = save_prefix.parent / (save_prefix.name + "_answer.txt")
    with open(answer_file, "wt") as f:
        for wavname, score in sorted(zip(wav_names, predict_mean_scores)):
            f.write(f"{wavname},{score}\n")
    wandb_logger.save(str(answer_file))

    predict_mean_scores = np.array(predict_mean_scores)
    true_mean_scores = np.array(true_mean_scores)
    predict_sys_mean_scores = np.array(
        [np.mean(scores) for scores in predict_sys_mean_scores.values()]
    )
    true_sys_mean_scores = np.array(
        [np.mean(scores) for scores in true_sys_mean_scores.values()]
    )

    # plot utterance-level histrogram
    plt.style.use("seaborn-deep")
    bins = np.linspace(1, 5, 40)
    plt.figure(2)

    plt.hist(
        [true_mean_scores, predict_mean_scores], bins, label=["true_mos", "predict_mos"]
    )
    plt.legend(loc="upper right")
    plt.xlabel("MOS")
    plt.ylabel("number")
    plt.show()
    figfile = str(save_prefix.parent / (save_prefix.name + "_distribution.png"))
    plt.savefig(figfile, dpi=150)
    if wandb_logger is not None:
        wandb_logger.log({f"{dataset_name}/distribution": wandb.Image(figfile)})
    plt.close()

    # utterance level scores
    MSE = np.mean((true_mean_scores - predict_mean_scores) ** 2)
    LCC = np.corrcoef(true_mean_scores, predict_mean_scores)[0][1]
    SRCC = scipy.stats.spearmanr(true_mean_scores, predict_mean_scores)[0]
    KTAU = scipy.stats.kendalltau(true_mean_scores, predict_mean_scores)[0]
    ep_scores += [MSE, LCC, SRCC, KTAU]
    if wandb_logger is not None:
        wandb_logger.log(
            {
                f"{dataset_name}/utt/MSE": MSE,
                f"{dataset_name}/utt/LCC": LCC,
                f"{dataset_name}/utt/SRCC": SRCC,
                f"{dataset_name}/utt/KTAU": KTAU,
            },
        )

    print(
        f"[UTTERANCE] {name} MSE: {float(MSE):.3f}, LCC: {float(LCC):.3f}, "
        f"SRCC: {float(SRCC):.3f}, KTAU: {float(KTAU):.3f}"
    )

    # plotting utterance-level scatter plot
    M = np.max([np.max(predict_mean_scores), 5])
    plt.figure(3)
    plt.scatter(
        true_mean_scores,
        predict_mean_scores,
        s=15,
        color="b",
        marker="o",
        edgecolors="b",
        alpha=0.20,
    )
    plt.xlim([0.5, M])
    plt.ylim([0.5, M])
    plt.xlabel("True MOS")
    plt.ylabel("Predicted MOS")
    plt.title(
        "Utt level LCC= {:.4f}, SRCC= {:.4f}, MSE= {:.4f}, KTAU= {:.4f}".format(
            LCC, SRCC, MSE, KTAU
        )
    )
    plt.show()
    figfile = str(save_prefix.parent / (save_prefix.name + "_utt_scatter_plot_utt.png"))
    plt.savefig(figfile, dpi=150)
    if wandb_logger is not None:
        wandb_logger.log({f"{dataset_name}/utt_scatter_plot": wandb.Image(figfile)})
    plt.close()

    # system level scores
    MSE = np.mean((true_sys_mean_scores - predict_sys_mean_scores) ** 2)
    LCC = np.corrcoef(true_sys_mean_scores, predict_sys_mean_scores)[0][1]
    SRCC = scipy.stats.spearmanr(true_sys_mean_scores, predict_sys_mean_scores)[0]
    KTAU = scipy.stats.kendalltau(true_sys_mean_scores, predict_sys_mean_scores)[0]
    ep_scores += [MSE, LCC, SRCC, KTAU]
    print(
        "[SYSTEM] {} MSE: {:.3f}, LCC: {:.3f}, SRCC: {:.3f}, KTAU: {:.3f}".format(
            name, float(MSE), float(LCC), float(SRCC), float(KTAU)
        )
    )
    if wandb_logger is not None:
        wandb_logger.log(
            {
                f"{dataset_name}/sys/MSE": MSE,
                f"{dataset_name}/sys/LCC": LCC,
                f"{dataset_name}/sys/SRCC": SRCC,
                f"{dataset_name}/sys/KTAU": KTAU,
            },
        )

    # plotting utterance-level scatter plot
    M = np.max([np.max(predict_sys_mean_scores), 5])
    plt.figure(3)
    plt.scatter(
        true_sys_mean_scores,
        predict_sys_mean_scores,
        s=15,
        color="b",
        marker="o",
        edgecolors="b",
    )
    plt.xlim([0.5, M])
    plt.ylim([0.5, M])
    plt.xlabel("True MOS")
    plt.ylabel("Predicted MOS")
    plt.title(
        "Sys level LCC= {:.4f}, SRCC= {:.4f}, MSE= {:.4f}, KTAU= {:.4f}".format(
            LCC, SRCC, MSE, KTAU
        )
    )
    plt.show()
    figfile = str(save_prefix.parent / (save_prefix.name + "_sys_scatter_plot_utt.png"))
    plt.savefig(figfile, dpi=150)
    if wandb_logger is not None:
        wandb_logger.log({f"{dataset_name}/sys_scatter_plot_utt": wandb.Image(figfile)})
    plt.close()

    return ep_scores


def update_results(name, result, ckpt_paths, result_path):
    result_path = Path(result_path)
    assert (
        not is_dirty_git_commit()
    ), "Commit and push to master before updating the results"
    if result_path.is_file():
        shutil.copyfile(result_path, result_path.with_suffix(".backup"))
        with open(result_path, "r", newline="") as csvfile:
            rows = list(csv.reader(csvfile))
        data = {row[0]: row[1:] for row in rows}
    else:
        data = {}
    data[name] = result + [ckpt_paths, get_git_commit()]
    rows = [[k] + v for k, v in data.items()]
    rows = sorted(rows)
    with open(result_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)
    # TODO save decoding commit and ckpt path


def compute_results(sysutt_mean_d, true_mean_scores, true_sys_mean_scores):
    """
    Heavily inspired from LDNet inference_for_voicemos.py script
    """
    predict_mean_scores = []
    predict_sys_mean_scores = defaultdict(list)

    for sysutt, mos in sysutt_mean_d.items():
        sysid, uttid = sysutt.split("-")
        predict_mean_scores.append(mos)
        predict_sys_mean_scores[sysid].append(mos)

    predict_mean_scores = np.array(predict_mean_scores)
    predict_sys_mean_scores = np.array(
        [np.mean(scores) for scores in predict_sys_mean_scores.values()]
    )

    # utterance level scores
    MSE = np.mean((true_mean_scores - predict_mean_scores) ** 2)
    LCC = np.corrcoef(true_mean_scores, predict_mean_scores)[0][1]
    SRCC = scipy.stats.spearmanr(true_mean_scores, predict_mean_scores)[0]
    KTAU = scipy.stats.kendalltau(true_mean_scores, predict_mean_scores)[0]
    utt_scores = [MSE, LCC, SRCC, KTAU]

    # system level scores
    MSE = np.mean((true_sys_mean_scores - predict_sys_mean_scores) ** 2)
    LCC = np.corrcoef(true_sys_mean_scores, predict_sys_mean_scores)[0][1]
    SRCC = scipy.stats.spearmanr(true_sys_mean_scores, predict_sys_mean_scores)[0]
    KTAU = scipy.stats.kendalltau(true_sys_mean_scores, predict_sys_mean_scores)[0]
    sys_scores = [MSE, LCC, SRCC, KTAU]

    return utt_scores + sys_scores
