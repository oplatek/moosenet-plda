import click
import torch
import sys
from lhotse.manipulation import combine
import numpy as np
import os
from moosenet.datasetup import get_noise_cuts
import shutil
from subprocess import DEVNULL, PIPE, run
import logging
import socket
from lhotse import CutSet
from lhotse.utils import ifnone
from pathlib import Path
import pytorch_lightning as pl
from typing import List, Dict, Any, Union, Optional, Tuple
from lhotse.utils import fix_random_seed
from pytorch_lightning import seed_everything


from moosenet.collate import CollateMOS
from moosenet.bin import moosenet
from moosenet.models import MOSModelMixin
from moosenet.dataloading import (
    get_test_dataloader,
    MOS_VC_TTSDataset,
)
from moosenet.datasetup import get_manifest
from moosenet.results import gen_results
from moosenet.utils import (
    parse_ckpt_path,
    is_dirty_git_commit,
    get_best_ckpts,
    get_MOSModel_cls,
)
from moosenet.average_checkpoints import load_avg_model


@moosenet.command()
@click.argument("ckpt_paths", nargs=-1, type=str)
@click.option("-s", "--splits", multiple=True)
@click.option("-e", "experiment_id", type=str, required=True)
@click.option(
    "-d",
    "--data_setup",
    type=str,
    default="voicemos_main1",
    help="Often used voicemos_main1 and voicemos_ood1_labeledonly",
)
@click.option("--outdir", type=str, default=None)
@click.option("--results_db", type=str, default=None)
@click.option("--decoder_init_lin", type=str, default=None)
@click.option(
    "--decoder_num_layers",
    type=int,
    default=1,
    help="For value 0 you need to set decoder_hidden_dim=768 for NO_finetune wav2vec_small ckpt",
)
@click.option("--decoder_hidden_dim", type=int, default=32)
@click.option("--limit_predict_batches", type=float, default=1.0)
@click.option("--limit_cuts", type=int, default=None)
@click.option(
    "--time_augment_train",
    type=int,
    default=1,
    help="Ratio how many time multiple data",
)
@click.option("--gpus", type=int, default=0)
@click.option(
    "--ckpt_best_from_dir",
    default=0,
    type=int,
    help="If 0 - do not use if 1+ select best ckpt from directory in ckpt_paths",
)
@click.option(
    "--ckpt_best_from_log",
    is_flag=True,
)
@click.option(
    "-w", "--dataloader_workers", default=4, help="Use 0 for main thread for debugging"
)
@click.option("-m", "--max_batch_duration", type=float, default=200, help="In seconds")
@click.option("--seed", type=int, default=42)
def infer(
    ckpt_paths: Tuple[str],
    splits: Tuple[str],
    experiment_id: str,
    data_setup: str,
    outdir: Optional[str],
    results_db: Optional[str],
    decoder_init_lin: Optional[str],
    decoder_num_layers: int,
    decoder_hidden_dim: int,
    limit_predict_batches: Union[float, int],
    limit_cuts: Optional[int],
    gpus: int,
    seed: int,
    time_augment_train: int,
    ckpt_best_from_dir: int,
    ckpt_best_from_log: int,
    dataloader_workers: int,
    max_batch_duration: float,
):
    """
    1. Run inference for given checkpoints.
    2. Saves metrics to results.csv file with the following columns
    ckpt_name, four_utterance_level_scores, same_four_scores_computed_per_system
        where scores are MSE, LCC, SRCC, KTAU
    3. Stores individual prediction per utterance
    4. Plot graphs
    """
    fix_random_seed(seed)
    seed_everything(seed, workers=True)
    logging.warning(f"Hostname {socket.gethostname()}")
    assert len(splits) > 0, "Provide any combination of splits train, dev, test"
    for split in splits:
        split in ["train", "dev", "test"], split
    results_db = Path(ifnone(results_db, "./results/results.csv")).resolve()
    outdir = Path(ifnone(outdir, "./results/ckpts")).resolve()

    if ckpt_paths == ("wav2vec_small",) or ckpt_paths == ("xlsr",):
        no_finetune = True
        ssl_name = ckpt_paths[0]
        ckpt_paths = []
    else:
        no_finetune = False

    if ckpt_best_from_dir:
        msg = f"{ckpt_best_from_dir=} -> assuming a single path to a existing directory but got {ckpt_paths}"
        assert len(ckpt_paths) == 1, msg
        ckpt_dir = Path(ckpt_paths[0])
        assert ckpt_dir.exists() and ckpt_dir.is_dir(), ckpt_dir
        ckpt_paths = get_best_ckpts(ckpt_dir, ckpt_best_from_dir)
        assert len(ckpt_paths) > 0, f"Found no ckpts in {ckpt_best_from_dir}"
    elif ckpt_best_from_log:
        msg = f"{ckpt_best_from_log=} -> assuming a single path to a log file but got {ckpt_paths}"
        assert len(ckpt_paths) == 1, msg
        with open(ckpt_paths[0]) as r:
            line = [ln for ln in r.readlines() if ln.startswith("BESTCKPT ")][0]
            ckpt_paths = [line[len("BESTCKPT ") :].strip()]

    for c in ckpt_paths:
        os.chmod(c, 0o444)  # readonly protection

    assert len(ckpt_paths) > 0 or no_finetune
    rn_wh_cs_list = [parse_ckpt_path(c) for c in ckpt_paths]
    if len(rn_wh_cs_list) == 0:
        assert no_finetune
        run_name = f"nofinetune_{ssl_name}"
        wandb_hash = "wbnotrain"
        ckpt_stem = "nockpt"
    elif len(rn_wh_cs_list) == 1:
        run_name, wandb_hash, ckpt_stem = rn_wh_cs_list[0]
    else:
        run_name = "AVG_" + "_".join([t[0] for t in rn_wh_cs_list])
        wandb_hash = "_".join([t[1] for t in rn_wh_cs_list])
        # ckpt_stem = "_".join([t[2] for t in rn_wh_cs_list])
        # Above cannot be used because it creates too long file names
        ckpt_stem = "nockpt"

    train_cuts, val_cuts, _val_egs_manifests, test_cuts = get_manifest(data_setup)

    if time_augment_train > 1:
        PERT_MIN, PERT_MAX = 0.9, 1.1
        c_perturb_tempo = np.arange(
            PERT_MIN, PERT_MAX, (PERT_MAX - PERT_MIN) * (time_augment_train - 1)
        )
        train_cuts = combine([train_cuts.perturb_tempo(f) for f in c_perturb_tempo])

    # For debugging use train cuts
    train_cuts = (
        train_cuts.subset(first=limit_cuts) if limit_cuts is not None else train_cuts
    )

    # see train script
    kwargs = dict(
        # we do not care about predicting noise labels (ATM)
        num_noise_labels=1,
    )
    if no_finetune:
        if ssl_name == "wav2vec_small":
            fairseq_model_path = (
                "../mos-finetune-ssl/fairseq-models/wav2vec_small.pt",
            )
        elif ssl_name == "xlsr":
            fairseq_model_path = (
                "../mos-finetune-ssl/fairseq-models/w2v_large_lv_fsh_swbd_cv.pt"
            )
        else:
            raise ValueError(f"Unknown {fairseq_model_path=}")
        Model = get_MOSModel_cls("SSL")
        kwargs = dict(
            fairseq_model_path=fairseq_model_path,
            decoder_num_layers=decoder_num_layers,
            decoder_hidden_dim=decoder_hidden_dim,
            decoder_init_lin=decoder_init_lin,
            num_noise_labels=1,  # dummy will not be used
        )
        model = Model(**kwargs)
    else:
        model = load_avg_model(ckpt_paths, **kwargs)
    model.add_embedding = True

    dataset = MOS_VC_TTSDataset.from_lhotse_cuts(
        score_strategy=CollateMOS.STRATEGY_MEAN,
        noise_cuts=get_noise_cuts("musan_noise"),
        betas=model.betas,
        dec_betas=model.dec_betas,
        model_name=model.name,
        noise_prob=1.0,
        include_cuts=True,
        # TODO change to true for logging/debugging
        include_audios=False,
    )

    exp_name = f"EMB_{data_setup}_{run_name}_{experiment_id}-{wandb_hash}_{ckpt_stem}"
    if limit_cuts is not None or limit_predict_batches != 1.0:
        exp_name = (
            f"{exp_name}_limitcuts_{limit_cuts}_limit_batches_{limit_predict_batches}"
        )
    outdir = outdir / exp_name
    logging.info(f"Saving results to {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    # shutil.copyfile(ckpt_path, outdir / Path(ckpt_path).name)

    # Wandb modes https://docs.wandb.ai/guides/track/launch#what-is-the-difference-between-wandb.init-modes  # noqa
    if limit_cuts is not None or limit_predict_batches != 1.0:
        wandb_mode = "disabled"
    else:
        wandb_mode = "online"
    logger = pl.loggers.wandb.WandbLogger(
        name=exp_name,
        entity="moosenet",
        project="voicemos_inf",
        save_dir=str(outdir),
        log_model=True,
        mode=wandb_mode,
    )
    with open(outdir / "launch_command.sh", "wt") as wt:
        wt.write("!/usr/bin/bash\n")
        wt.write(f"cd '{os.getcwd()}'\n")
        wt.write(" ".join(sys.argv) + "\n")

    trainer = pl.Trainer(
        gpus=gpus,
        limit_predict_batches=limit_predict_batches,
        strategy="ddp" if gpus > 1 else None,  # PL 1.5.+
        logger=logger,
    )

    cuts_dir = {
        "test": test_cuts,
        "dev": val_cuts,
        "train": train_cuts,
    }

    for split in splits:
        cuts = cuts_dir[split]
        logging.info(f"Predicting {split}")
        test_loader = get_test_dataloader(
            cuts,
            dataset,
            dataloader_workers,
            max_batch_duration=max_batch_duration,
            prefetch_factor=2,
        )
        # see models.MOSModelMixin predict_step method for outputs specs
        predictions: List[Dict[Any]] = trainer.predict(
            model=model,
            dataloaders=[test_loader],
            return_predictions=True,
        )
        torch.save(predictions, outdir / f"{split}_nn_predictions.pth")
        logging.info(f"{split} predictions saved. Updating the stats.")

        gen_results(
            predictions,
            outdir,
            exp_name,
            f"{data_setup}_{split}",
            logger.experiment,
        )
        logging.info(f"{split} results and stats saved. Updating the stats.")
    logging.info("Successfully finished!")
    print(f"OUTDIR {outdir}", flush=True)
    print("SUCCESSFULLY FINISHED", flush=True)
