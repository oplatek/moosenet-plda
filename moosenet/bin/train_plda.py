import click
import torch
import wandb
import os
from lhotse.utils import fix_random_seed
from pytorch_lightning import seed_everything
from moosenet.datasetup import get_noise_cuts
import shutil
from subprocess import DEVNULL, PIPE, run
import logging
from lhotse import CutSet
from lhotse.utils import ifnone
from pathlib import Path
import pytorch_lightning as pl
from typing import List, Dict, Any, Union, Optional, Tuple
from plda import Classifier


from moosenet.collate import CollateMOS
from moosenet.bin import moosenet
from moosenet.models import MOSModelMixin
from moosenet.dataloading import (
    get_test_dataloader,
    MOS_VC_TTSDataset,
)
from moosenet.datasetup import get_manifest
from moosenet.results import gen_results, upload_predictions
from moosenet.utils import parse_ckpt_path, is_dirty_git_commit
from moosenet.average_checkpoints import load_avg_model
from moosenet.plda_utils import fit_plda
from moosenet.plda_utils import infer_plda

# from moosenet.plda_utils import infer_pldaXxY as infer_plda


@moosenet.command()
@click.option("-d", "emb_dir", default=None)
@click.option("-f", "--emb_dir_from_log", default=None)
@click.option("-e", "experiment_id", type=str, required=True)
@click.option("--use_other_datasetup_eval", is_flag=True)
@click.option("--weight_bins/--argmax_bins", default=True)
@click.option("--skip_pca", is_flag=True)
@click.option("--seed", type=int, default=42)
@click.option("--multiply_data", type=int, default=1)
@click.option("--load_pca_from_model", default=None)
@click.option("--plda_feat_dim", default=None, type=int)
@click.option("--n_bins", default=32, type=int)
@click.option(
    "--add_gauss_noise",
    default=0.0,
    type=float,
    help="add gauss noise to avoid dependent rows for PCA decomposition. If PCA crashes with 0.0, start with 0.001.",
)
def train_plda(
    emb_dir: str,
    emb_dir_from_log: str,
    experiment_id: str,
    use_other_datasetup_eval: bool,
    weight_bins: bool,
    skip_pca: bool,
    load_pca_from_model: str,
    plda_feat_dim: Optional[int],
    n_bins: int,
    add_gauss_noise: float,
    seed: int,
    multiply_data: int,
):
    fix_random_seed(seed)
    seed_everything(seed, workers=True)
    if emb_dir is None:
        assert emb_dir_from_log is not None
        with open(emb_dir_from_log) as r:
            line = [ln for ln in r.readlines() if ln.startswith("OUTDIR ")][0]
            emb_dir = line[len("OUTDIR ") :].strip()
    else:
        assert emb_dir_from_log is None
    emb_dir = Path(emb_dir)
    emb_dir_name: str = f"{emb_dir.stem}"

    # first version started with emb second with EMB
    assert emb_dir_name.lower().startswith("emb_"), emb_dir_name
    if emb_dir_name[4 : 4 + len("voicemos_main1")] == "voicemos_main1":
        data_setup = "voicemos_main1"
        eval_data_setup = "voicemos_ood1_labeledonly"
    elif (
        emb_dir_name[4 : 4 + len("voicemos_ood1_labeledonly")]
        == "voicemos_ood1_labeledonly"
    ):
        data_setup = "voicemos_ood1_labeledonly"
        eval_data_setup = "voicemos_main1"
    else:
        raise ValueError(f"Uknonwn datasetup for emb dir {emb_dir}")

    if use_other_datasetup_eval:
        eval_emb_dir = Path(str(emb_dir).replace(data_setup, eval_data_setup))
        assert eval_emb_dir.exists(), f"The path does not exits {eval_emb_dir=}"
    else:
        eval_emb_dir = None

    train_nn_predictions = torch.load(emb_dir / "train_nn_predictions.pth")
    if skip_pca:
        pca = "skip"
    elif load_pca_from_model:
        assert not skip_pca
        pca = Classifier.load_model(load_pca_from_model).model.pca
        assert pca is not None
    else:
        pca = None  # new PCA will be trained

    plda_model, bins_borders = fit_plda(
        train_nn_predictions,
        n_bins,
        feat_dim=plda_feat_dim,
        pca=pca,
        multiply_data=multiply_data,
        add_gauss_noise=add_gauss_noise,
    )
    plda_model.save_model(emb_dir / "plda.pickle")

    exp_name = f"{experiment_id}_{emb_dir_name}"

    test_nn_predictions = torch.load(emb_dir / "test_nn_predictions.pth")
    logger = wandb.init(
        name=exp_name,
        entity="moosenet",
        project="voicemos_plda_train",
        dir=str(emb_dir),
    )
    test_plda_predictions = infer_plda(plda_model, bins_borders, test_nn_predictions)
    gen_results(
        test_nn_predictions,
        emb_dir,
        exp_name,
        f"NN_{data_setup}_test",
        logger,
    )
    gen_results(
        test_plda_predictions,
        emb_dir,
        exp_name,
        f"PLDA_{data_setup}_test",
        logger,
    )
    del test_nn_predictions
    del test_plda_predictions

    train_plda_predictions = infer_plda(
        plda_model, bins_borders, train_nn_predictions, weight_bins=weight_bins
    )
    gen_results(
        train_nn_predictions,
        emb_dir,
        exp_name,
        f"NN_{data_setup}_train",
        logger,
    )
    gen_results(
        train_plda_predictions,
        emb_dir,
        exp_name,
        f"PLDA_{data_setup}_train",
        logger,
    )
    del train_nn_predictions
    del train_plda_predictions

    dev_nn_predictions = torch.load(emb_dir / "dev_nn_predictions.pth")
    dev_plda_predictions = infer_plda(plda_model, bins_borders, dev_nn_predictions)
    gen_results(
        dev_nn_predictions,
        emb_dir,
        exp_name,
        f"NN_{data_setup}_dev",
        logger,
    )
    gen_results(
        dev_plda_predictions,
        emb_dir,
        exp_name,
        f"PLDA_{data_setup}_dev",
        logger,
    )
    del dev_nn_predictions
    del dev_plda_predictions

    if eval_emb_dir is None:
        return
    ###############################
    assert eval_emb_dir is not None
    eval_train_nn_predictions = torch.load(eval_emb_dir / "train_nn_predictions.pth")
    eval_train_plda_predictions = infer_plda(
        plda_model, bins_borders, eval_train_nn_predictions, weight_bins=weight_bins
    )
    gen_results(
        eval_train_nn_predictions,
        emb_dir,
        exp_name,
        f"NN_{eval_data_setup}_train",
        logger,
    )
    gen_results(
        eval_train_plda_predictions,
        emb_dir,
        exp_name,
        f"PLDA_{eval_data_setup}_train",
        logger,
    )
    del eval_train_nn_predictions
    del eval_train_plda_predictions

    eval_dev_nn_predictions = torch.load(eval_emb_dir / "dev_nn_predictions.pth")
    eval_dev_plda_predictions = infer_plda(
        plda_model, bins_borders, eval_dev_nn_predictions
    )
    gen_results(
        eval_dev_nn_predictions,
        emb_dir,
        exp_name,
        f"NN_{eval_data_setup}_dev",
        logger,
    )
    gen_results(
        eval_dev_plda_predictions,
        emb_dir,
        exp_name,
        f"PLDA_{eval_data_setup}_dev",
        logger,
    )
    del eval_dev_nn_predictions
    del eval_dev_plda_predictions

    eval_test_nn_predictions = torch.load(eval_emb_dir / "test_nn_predictions.pth")
    eval_test_plda_predictions = infer_plda(
        plda_model, bins_borders, eval_test_nn_predictions
    )
    upload_predictions(
        eval_test_nn_predictions,
        emb_dir,
        exp_name,
        f"NN_{eval_data_setup}_test",
        logger,
    )
    upload_predictions(
        eval_test_plda_predictions,
        emb_dir,
        exp_name,
        f"PLDA_{eval_data_setup}_test",
        logger,
    )

    del eval_test_nn_predictions
    del eval_test_plda_predictions
