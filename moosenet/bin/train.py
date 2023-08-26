import click
import socket
from pathlib import Path
from typing import List, Dict, Any
from moosenet.results import gen_results, upload_predictions
from numpy import arange
import wandb
from math import ceil
import json
from copy import deepcopy
import datetime
import pytorch_lightning as pl
import logging
from lhotse import CutSet
from lhotse.utils import fix_random_seed
from pytorch_lightning import seed_everything
from moosenet import exp_dir, data_dir
from moosenet.utils import SetSamplerEpoch, get_MOSModel_cls, get_best_ckpts
from moosenet.datasetup import get_manifest, get_noise_cuts, load_annotator_stats
from moosenet.average_checkpoints import load_avg_model
from moosenet.bin import moosenet
from moosenet.dataloading import (
    get_train_dataloader,
    get_val_dataloader,
    get_test_dataloader,
    MOS_VC_TTSDataset,
)
from moosenet.collate import CollateMOS
from moosenet.models import DEFAULT_BETAS, DEFAULT_DEC_BETAS
from lhotse.dataset.cut_transforms import PerturbVolume, PerturbTempo


@moosenet.command()
@click.option("-e", "experiment_id", type=int, required=True)
@click.option(
    "-d",
    "description",
    help="Argument for the sake that the cmd and this argument is logged to wandb",
)
@click.option("--fast_dev_run", is_flag=True)
@click.option(
    "--watch_model", default="none", help="grads_topology, grads_hist_topology or none"
)
@click.option("--frame_loss/--skip_frame_loss", default=True)
@click.option("--dummy_collate", is_flag=True)
@click.option("--gpus", type=int, default=1)
@click.option(
    "--vizualize_every_epoch",
    type=int,
    default=-1,
    help="Specify zero or negative number to avoid vizualization logging",
)
@click.option("--vizualize_ground_truth", is_flag=True)
@click.option("--use_deducted_mos_scores", is_flag=True)
@click.option("--limit_train_batches", type=float, default=1.0)
@click.option("--limit_val_batches", type=float, default=1.0)
@click.option("--noise_prob", type=float, default=1.0)
@click.option("--positive_prob", type=float, default=0.0)
@click.option("--train_ratio", type=float, default=1.0)
@click.option(
    "--ckpt_best_from_dir",
    default=0,
    type=int,
    help="If 0 - do not use if 1+ select best ckpt from directory in ckpt_paths",
)
@click.option(
    "-w", "--dataloader_workers", default=24, help="Use 0 for main thread for debugging"
)
@click.option("--max_steps", type=int, default=10e6)
@click.option(
    "-r", "--run", default=datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
)
@click.option("--seed", type=int, default=42)
@click.option("--prefetch_factor", type=int, default=4)
@click.option("--num_buckets", type=int, default=20)
@click.option("--precision", type=int, default=16)
@click.option("--patience", type=int, default=30)
@click.option(
    "--detect_anomaly",
    is_flag=True,
    help="https://pytorch.org/docs/stable/autograd.html#anomaly-detection",
)
@click.option("-d", "--data_setup", type=str, default="voicemos_main1")
@click.option(
    "--betas",
    default=",".join(f"{k},{v}" for k, v in DEFAULT_BETAS.items()),
)
@click.option(
    "--dec_betas",
    default=",".join(f"{k},{v}" for k, v in DEFAULT_DEC_BETAS.items()),
)
@click.option("--clip_tau", type=float, default=0.0)
@click.option("--contrast_margin", type=float, default=0.2)  # 0.2 too low set it to 0.5
@click.option("--triplet_loss_margin", type=float, default=0.05)
@click.option(
    "--triplet_emb",
    default="loss",
    help="loss or emb for dec_final_out or mos_final layers",
)
@click.option("--triplet_mine_pos_margin", type=float, default=0.1)
@click.option("--triplet_mine_neg_margin", type=float, default=0.5)
@click.option(
    "--load_weights",
    type=str,
    default=None,
    multiple=True,
    help=(
        "specify ckpt from which to just load weights. Be careful to set optimizers etc manually."
        "Mutually exclusive with train_resume."
    ),
)
@click.option(
    "--train_resume",
    type=str,
    default=None,
    help=(
        "Specify ckpt from which to resume training. Mutually exclusive with load_weights."
        "train_resume expect exact match of state_dict"
    ),
)
@click.option("-n", "--noise_data", type=str, default="musan_noise", help="musan_noise")
@click.option(
    "-m",
    "--model",
    default="SSL",
    help="SSL is the only model used in the paper",
)
@click.option("--freeze_ssl", is_flag=True)
@click.option("--without_augmentation", is_flag=True)
@click.option("--freeze_decoder", is_flag=True)
@click.option(
    "--fairseq_model_path",
    default="../mos-finetune-ssl/fairseq-models/wav2vec_small.pt",
    help="Path to pretrained fairseq checkpoint",
)
@click.option(
    "--ckpt_metric",
    type=str,
    default="val/final_mos_scc/dataloader_idx_0",
    help="Use val/final_mos_mse/dataloader_idx_0 or val/final_mos_scc/dataloader_idx_0",
)
@click.option("--n_ckpts", default=4)
@click.option(
    "--mos_projection",
    type=str,
    default="clip_range",
    help="clip_range or clip_range_noeps or linear",
)
@click.option(
    "-o",
    "--optimizer",
    type=str,
    default="lamb",
    help="lamb, adam, SGD, etc",
)
@click.option(
    "-s",
    "--scheduler",
    type=str,
    default="warmup",
    help="warmup_step, warmup, step, None",
)
@click.option(
    "--score_strategy",
    type=str,
    default=CollateMOS.STRATEGY_MEAN,
    help=f"{CollateMOS.STRATEGY_MODES}",
)
@click.option("--max_batch_duration", type=float, default=800)
@click.option("--min_cut_duration", type=float, default=1.0)
@click.option("--max_cut_duration", type=float, default=12.0)
@click.option("--learning_rate", type=float, default=0.001)
@click.option("--weight_decay", type=float, default=0.0001)
# TODO delete if verified the weight_decay could be used as recommended
# @click.option("--weight_decay", type=float, default=0.01)
@click.option(
    "-lrh",
    "--learning_rate_halve_steps",
    type=int,
    default=1000 * 30,
    show_default=True,
    help="Reduce LR by half after the steps, Used wth lamb_stepLR combination",
)
@click.option(
    "--warmup_steps",
    type=int,
    default=1500,
    show_default=True,
    help="Warmup steps for adam. Used with adam_warmupLinDecay",
)
@click.option("--conformer_layer_input_dim", type=int, default=512)
@click.option("--conformer_num_layers", type=int, default=12)
@click.option("--decoder_num_layers", type=int, default=1)
@click.option("--decoder_hidden_dim", type=int, default=768)
@click.option("--decoder_dropout", type=float, default=0.3)
@click.option("--listener_emb_size", type=int, default=0)
def train(
    train_resume: str,
    betas: str,
    dec_betas: str,
    load_weights: str,
    model: str,
    fairseq_model_path: str,
    freeze_ssl: bool,
    without_augmentation: bool,
    freeze_decoder: bool,
    data_setup: str,
    noise_data: str,
    score_strategy: str,
    conformer_layer_input_dim: int,
    conformer_num_layers: int,
    decoder_num_layers: int,
    decoder_hidden_dim: int,
    decoder_dropout: float,
    listener_emb_size: int,
    noise_prob: float,
    positive_prob: float,
    train_ratio: float,
    clip_tau: float,
    contrast_margin: float,
    triplet_loss_margin: float,
    triplet_emb: str,
    triplet_mine_pos_margin: float,
    triplet_mine_neg_margin: float,
    limit_train_batches: float,
    limit_val_batches: float,
    fast_dev_run: bool,
    watch_model: str,
    frame_loss: bool,
    dummy_collate: bool,
    gpus: int,
    vizualize_every_epoch: int,
    vizualize_ground_truth: bool,
    use_deducted_mos_scores: bool,
    ckpt_best_from_dir: int,
    dataloader_workers: int,
    max_steps: int,
    experiment_id: int,
    description: str,
    run: str,
    seed: int,
    prefetch_factor: int,
    num_buckets: int,
    max_batch_duration: float,
    min_cut_duration: float,
    max_cut_duration: float,
    ckpt_metric: str,
    n_ckpts: int,
    mos_projection: str,
    optimizer: str,
    scheduler: str,
    learning_rate: float,
    weight_decay: float,
    learning_rate_halve_steps: int,
    warmup_steps: int,
    precision: int,
    patience: int,
    detect_anomaly: bool,
):
    fix_random_seed(seed)
    seed_everything(seed, workers=True)
    logging.warning(f"Hostname {socket.gethostname()}")
    prefetch_factor = 2 if dataloader_workers == 0 else prefetch_factor

    assert listener_emb_size <= 1 or score_strategy in [
        CollateMOS.STRATEGY_RANDOM_AVG,
        CollateMOS.STRATEGY_RANDOM,
    ], "Listener Dependent modeling need individual human scores"

    betas = betas.split(",")
    betas = dict((n, float(v)) for n, v in zip(betas[::2], betas[1::2]))
    assert betas.keys() == DEFAULT_BETAS.keys(), f"{betas} vs {DEFAULT_BETAS}"
    dec_betas = dec_betas.split(",")
    dec_betas = dict((n, float(v)) for n, v in zip(dec_betas[::2], dec_betas[1::2]))
    assert list(dec_betas.keys()) == list(
        DEFAULT_DEC_BETAS.keys()
    ), f"{dec_betas} vs {DEFAULT_DEC_BETAS}"
    assert sum(dec_betas.values()) > 0, "We should train something"

    if ckpt_metric == "val/final_mos_mse/dataloader_idx_0":
        ckpt_metric_mode = "min"
        ckpt_metric_name = "final_mos_mse"
    elif ckpt_metric == "val/final_mos_scc/dataloader_idx_0":
        ckpt_metric_mode = "max"
        ckpt_metric_name = "final_mos_scc"
    else:
        raise ValueError(f"Unsupported ckpt_metric {ckpt_metric}")

    logging.warning(
        f"Training experiment with ID started: {experiment_id}\nDesc: {description}"
    )
    assert not (
        load_weights is not None and train_resume is not None
    ), "Do not combine load_weights and train_resume"
    assert (
        train_resume is None or not fast_dev_run
    ), "fast_dev_run sets max_epoch 1 to Trainer and resume from training is typically done with epoch > 1"
    if load_weights or train_resume:
        # Not loading from from pretrained fairseq model - the weights are fine tuned in our saved ckpt
        fairseq_model_path = None

    assert precision == 16 or precision == 32, f"Fix precision {precision} to 16 or 32"

    if "main1" in data_setup:
        wandb_project = "voicemos_main"
    elif "ood1" in data_setup:
        wandb_project = "voicemos_ood"
    else:
        raise ValueError(f"Unkown datasetup {data_setup} for logging in wandb")
    train_cuts, val_cuts, val_egs_manifests, test_cuts = get_manifest(
        data_setup, train_ratio
    )
    # get noise cuts if used by loss
    if (
        betas["snr"] > 0.0
        or betas["noise_label"] > 0.0
        or dec_betas["snr"] > 0.0
        or dec_betas["noise_label"] > 0.0
        or dec_betas["stoi"] > 0.0
        # or dec_betas["consist_mos"] > 0.0
        or dec_betas["mcd"] > 0.0
    ):
        noise_cuts = get_noise_cuts(noise_data)  # eager cutset in memory
        assert (
            noise_prob == 1.0
        ), f"{noise_prob} vs we showed the more feedback the better"
    else:
        noise_cuts = CutSet.from_cuts([])  # empty noise cutset
        assert noise_prob == 0.0, str(noise_prob)

    if not frame_loss:
        assert all(v <= 0.0 for v in betas.values()), str(betas)

    dataset = MOS_VC_TTSDataset.from_lhotse_cuts(
        score_strategy,
        noise_cuts,
        betas,
        dec_betas,
        model,
        noise_prob=noise_prob,
        wave_transforms=[]
        if without_augmentation
        else [
            # preserve_id keeps the cut ids the same - DANGEROUS - check dataset that it does not break things
            # TODO check that SSL models do not normalize volume
            PerturbVolume(scale_low=0.5, scale_high=2.0, p=0.8, preserve_id=True),
            PerturbTempo(
                factors=list(arange(0.9, 1.08, 0.001)), p=1.0, preserve_id=True
            ),
            # TODO pitch via torchaudio. Add AugmentFn to be call on CutSet to add augment_fn to each cut.
        ],
        positive_cuts=positive_prob,
    )

    train_dataloader = get_train_dataloader(
        train_cuts,
        dataset,
        dataloader_workers,
        prefetch_factor,
        num_buckets,
        max_batch_duration,
        min_cut_duration,
        max_cut_duration,
    )

    val_dataset = deepcopy(dataset)
    val_dataset.wave_transforms = []
    val_dataloader = get_val_dataloader(
        val_cuts,
        val_dataset,
        dataloader_workers,
        prefetch_factor,
        # compensate less filtering in validation set by larger batches
        max_batch_duration / 2.0,
        # TODO change these parameters only after comparison experiment
        # min_cut_duration / 2.0,
        # max_cut_duration * 1.2,
        min_cut_duration,
        max_cut_duration,
    )

    egs_dataset = deepcopy(val_dataset)
    egs_dataset.include_cuts = True
    egs_dataset.include_audios = True
    val_egs_dataloaders = [
        get_val_dataloader(
            egs_cuts,
            egs_dataset,
            dataloader_workers,
            prefetch_factor,
            max_batch_duration / 4.0,
            min_cut_duration / 2.0,
            max_batch_duration * 1.5,
        )
        for egs_cuts in val_egs_manifests
    ]

    run = f"{experiment_id}_{run}"
    wandb_dir = exp_dir / run
    wandb_dir.mkdir(exist_ok=True, parents=True)

    wandb_logger = pl.loggers.wandb.WandbLogger(
        name=f"{run}",
        entity="moosenet",
        project=wandb_project,
        save_dir=str(wandb_dir),
        log_model=False,
    )

    limit_train_batches = (
        int(limit_train_batches) if limit_train_batches > 1.0 else limit_train_batches
    )
    limit_val_batches = (
        int(limit_val_batches) if limit_val_batches > 1.0 else limit_val_batches
    )

    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        auto_insert_metric_name=False,
        save_last=False,
        # val_dataloader runs on whole epoch
        monitor=ckpt_metric,
        mode=ckpt_metric_mode,
        save_top_k=n_ckpts,
        filename="E{epoch:02d}-val_final_mos_"
        + ckpt_metric_name
        + "_{"
        + ckpt_metric
        + ":.4f}",
    )
    trainer = pl.Trainer(
        fast_dev_run=fast_dev_run,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        log_every_n_steps=2,
        gpus=gpus,
        max_steps=max_steps,
        detect_anomaly=detect_anomaly,
        precision=precision if gpus > 0 else 32,
        sync_batchnorm=True if gpus > 1 else False,
        strategy="ddp" if gpus > 1 else None,  # PL 1.5.+
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        replace_sampler_ddp=False,
        callbacks=[
            checkpoint_cb,
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            SetSamplerEpoch(),
            pl.callbacks.EarlyStopping(
                monitor=ckpt_metric,
                mode=ckpt_metric_mode,
                patience=patience,
            ),
        ],
        logger=wandb_logger,
    )

    if not fast_dev_run:
        # This actually creates the run
        # Define how summary of the metric should behave in Wandb
        wandb_logger.experiment.define_metric(
            "val/final_mos_mse/dataloader_idx_0", summary="min"
        )
        wandb_logger.experiment.define_metric(
            "val/final_mos_pcc/dataloader_idx_0", summary="max"
        )
        wandb_logger.experiment.define_metric(
            "val/final_mos_scc/dataloader_idx_0", summary="max"
        )

    Model = get_MOSModel_cls(model)
    # Models accept parameters by kwargs and ignoring unused parameter
    # Dangerous - type precisely ;-)
    kwargs = dict(
        frame_loss=frame_loss,
        fairseq_model_path=fairseq_model_path,
        freeze_ssl=freeze_ssl,
        freeze_decoder=freeze_decoder,
        vizualize_every_epoch=vizualize_every_epoch,
        vizualize_ground_truth=vizualize_ground_truth,
        use_deducted_mos_scores=use_deducted_mos_scores,
        clip_tau=clip_tau,
        contrast_margin=contrast_margin,
        triplet_emb=triplet_emb,
        triplet_loss_margin=triplet_loss_margin,
        triplet_mine_pos_margin=triplet_mine_pos_margin,
        triplet_mine_neg_margin=triplet_mine_neg_margin,
        fbank_dim=dataset.fbanks_dim,
        optimizer=optimizer,
        scheduler=scheduler,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_ctc_trnphones=len(dataset.phones_trn_idx2p),
        num_noise_labels=len(dataset.noise_labels_idx2n),
        learning_rate_halve_steps=learning_rate_halve_steps,
        warmup_steps=warmup_steps,
        conformer_num_layers=conformer_num_layers,
        conformer_layer_input_dim=conformer_layer_input_dim,
        decoder_num_layers=decoder_num_layers,
        decoder_hidden_dim=decoder_hidden_dim,
        decoder_dropout=decoder_dropout,
        listener_emb_size=listener_emb_size,
        num_listeners=dataset.num_listeners,
        mos_projection=mos_projection,
        resume_from_checkpoint=train_resume,
        betas=betas,
        dec_betas=dec_betas,
    )
    if train_resume is not None:
        logging.warning(
            "Not sure how PL handles overloading the hyperparams and replacing them from ckpt - be carefull"
        )
        logging.warning(
            "Train resumes the model but does not restore the original dataloading. "
            "The collations, datasetup, sampler, etc ... the whole data loading pipeline "
            "needs to match to have an exact continuation of the training."
        )
    # Note: train_resume and load_weights are mutually exclusive
    ckpt_paths = load_weights if load_weights is not None else [train_resume]
    if ckpt_best_from_dir:
        msg = f"{ckpt_best_from_dir=} -> assuming a single path to a existing directory but got {ckpt_paths}"
        assert len(ckpt_paths) == 1, msg
        ckpt_dir = Path(ckpt_paths[0])
        assert ckpt_dir.exists() and ckpt_dir.is_dir(), ckpt_dir
        ckpt_paths = get_best_ckpts(ckpt_dir, ckpt_best_from_dir)
        assert len(ckpt_paths) > 0, f"Found no ckpts in {ckpt_best_from_dir}"

    model = (
        load_avg_model(ckpt_paths, **kwargs, prefer_ckpt=False)
        if ckpt_paths
        else Model(**kwargs)
    )

    if watch_model == "grads_topology":
        wandb_logger.watch(model)
    elif watch_model == "grads_hist_topology":
        wandb_logger.watch(model, log="all")
    else:
        assert watch_model == "none", watch_model

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=[val_dataloader] + val_egs_dataloaders,
    )

    # see models.MOSModelMixin predict_step method for outputs specs
    results_dataset = MOS_VC_TTSDataset.from_lhotse_cuts(
        score_strategy=CollateMOS.STRATEGY_MEAN,  # at inference the strategy has to mean always!
        # We will use clean cuts part of the batch in trainer.predict this is just me lazy
        noise_cuts=get_noise_cuts("musan_noise"),
        betas=model.betas,
        dec_betas=model.dec_betas,
        model_name=model.name,
        noise_prob=1.0,
        include_cuts=True,
    )
    best_model = load_avg_model([checkpoint_cb.best_model_path], prefer_ckpt=True)

    logging.info("evaluating best model on test set")
    test_predictions: List[Dict[Any]] = trainer.predict(
        model=best_model,
        dataloaders=[
            get_test_dataloader(
                test_cuts,
                results_dataset,
                dataloader_workers,
                prefetch_factor,
                max_batch_duration=max_batch_duration,
            )
        ],
        return_predictions=True,
    )
    gen_results(
        test_predictions,
        trainer.logger.experiment.dir,
        run,
        f"{data_setup}_test",
        wandb_logger.experiment,
    )
    upload_predictions(
        test_predictions,
        trainer.logger.experiment.dir,
        run,
        f"{data_setup}_test",
        wandb_logger.experiment,
    )
    logging.info("Test predictions and stats saved.")

    logging.info("evaluating best model on validation set")
    val_predictions: List[Dict[Any]] = trainer.predict(
        model=best_model,
        dataloaders=[
            get_test_dataloader(
                val_cuts,
                results_dataset,
                dataloader_workers,
                prefetch_factor,
                max_batch_duration=max_batch_duration,
            )
        ],
        return_predictions=True,
    )
    gen_results(
        val_predictions,
        trainer.logger.experiment.dir,
        run,
        f"{data_setup}_val",
        wandb_logger.experiment,
    )
    upload_predictions(
        val_predictions,
        trainer.logger.experiment.dir,
        run,
        f"{data_setup}_val",
        wandb_logger.experiment,
    )
    logging.info("Dev predictions and stats saved.")
    print(f"BESTCKPT {checkpoint_cb.best_model_path}", flush=True)
    print("SUCCESSFULLY FINISHED", flush=True)
