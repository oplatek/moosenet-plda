import logging
from torch import index_select
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from lhotse.utils import ifnone
import torch
import torchaudio
import pytorch_lightning as pl
from torch import nn
from torch_optimizer import Lamb
from torchmetrics import MeanSquaredError, PearsonCorrCoef, SpearmanCorrCoef
from pytorch_lightning.trainer.states import RunningStage

from moosenet.utils import mask_encoder_outputs, get_MOSModel_cls, length_to_mask
from moosenet.modules import (
    MOSScoreProjection,
    ScoreProjection,
    BiLSTMDecoder,
    MaxAvgPoolNN,
    LabelsProjection,
    WarmupStepLR,
    WarmupLR,
)
from moosenet.panns_modules import Cnn14
from moosenet.loss import (
    Regress_CTC_NoiseCE_Loss,
    CombinedFinalLoss,
    mine_triplets,
)
from moosenet.collate import (
    CollateNoiseLabels,
    CollateSTOI_MCD,
    CollateMCD,
    CollateMOS,
    CollatePitch,
    CollatePhonesFromTrn,
    CollateFbank,
    PadAudio,
)
from moosenet.logs import (
    training_step_end,
    validation_step_end,
    step_vizualize,
    vizualize_ground_truth,
)

SUBSAMPLING_FACTOR = 4  # caused by Conformer subsampling - hack

DEFAULT_BETAS = {
    "ctc_phntrn": 0.0,
    "moss": 0.0,
    "pitch": 0.0,
    "snr": 0.0,
    "stoi": 0.0,
    "noise_label": 0.0,
}

DEFAULT_DEC_BETAS = {
    "mos_final": 1.0,
    "var_mos_final": 0.0,
    "snr": 0.0,
    "stoi": 0.0,
    "noise_label": 0.0,
    "consist_mos": 0.0,
    "mcd": 0.0,
    "contrast_mos": 0.0,
}


class MOSModelMixin(pl.LightningModule, ABC):
    MODEL_CLASS_NAMES = ("SSL", "ConformerFinalProjection", "ConformerFrameProjection")

    def __init__(
        self,
        *,
        name: str,  # model class name
        optimizer: str = "lamb",
        scheduler: str = "warmup",
        learning_rate: float = 0.01,
        weight_decay: float = 0.01,
        learning_rate_halve_steps: int = 16000,
        warmup_steps: int = 16000,
        vizualize_every_epoch: int = -1,
        vizualize_ground_truth: bool = False,
        add_embedding: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert name in self.MODEL_CLASS_NAMES, f"{name} vs {self.MODEL_CLASS_NAMES}"
        self.name = name
        self.vizualize_every_epoch = vizualize_every_epoch
        self.vizualize_ground_truth = vizualize_ground_truth

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.learning_rate_halve_steps = learning_rate_halve_steps

        # self.train_frame_mos_mse = MeanSquaredError()
        self.val_frame_mos_mse = MeanSquaredError()
        self.val_frame_mos_pearson = PearsonCorrCoef()
        self.val_frame_mos_spearman = SpearmanCorrCoef()
        # TODO make optionnal with refactoring logging into mixins
        # and splitting this class to optimizer and logging mixin
        # self.train_final_mos_mse = MeanSquaredError()
        self.val_final_mos_mse = MeanSquaredError()
        self.val_final_mos_pearson = PearsonCorrCoef()
        self.val_final_mos_spearman = SpearmanCorrCoef()

        self.add_embedding = add_embedding  # hacky way how to produce embedding
        self.save_hyperparameters()

    @property
    def betas(self):
        # set betas to 0.0 ie not using the losses if not present in the model loss
        return (
            self.loss.betas
            if hasattr(self, "loss")
            else dict((k, 0.0) for k in DEFAULT_BETAS)
        )

    @property
    def dec_betas(self):
        # set betas to 0.0 ie not using the losses if not present in the model loss
        return (
            self.dec_loss.betas
            if hasattr(self, "dec_loss")
            else dict((k, 0.0) for k in DEFAULT_DEC_BETAS)
        )

    def configure_optimizers(self):
        if self.optimizer == "lamb":
            optimizer = Lamb(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "adam":
            betas = (
                0.9,
                0.98,
            )  # coefficients for running avgs of gradient and its square
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                betas=betas,
                eps=1e-8,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer == "SGD":
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.learning_rate, momentum=0.9
            )
        else:
            raise ValueError("Unknown otimizer {self.optimizer}")

        if self.scheduler == "None":
            scheduler = None
        elif self.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, self.learning_rate_halve_steps, gamma=0.5
            )
        elif self.scheduler == "warmup":
            scheduler = WarmupLR(optimizer, self.warmup_steps)
        elif self.scheduler == "warmup_step":
            scheduler = WarmupStepLR(
                optimizer, self.warmup_steps, self.learning_rate_halve_steps, gamma=0.5
            )
        else:
            raise ValueError(f"Uknown scheduler {self.scheduler}")

        if scheduler is not None:
            # support only single scheduler
            schedulers = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        else:
            schedulers = []

        return [optimizer], schedulers

    def training_step(self, batch, batch_idx):
        outputs = self._step(batch)
        outputs["num_clean_cuts"] = batch["num_clean_cuts"]
        outputs["num_noisy_cuts"] = batch["num_noisy_cuts"]
        outputs["num_pos_cuts"] = batch["num_pos_cuts"]
        outputs["num_cuts"] = batch["num_cuts"]
        return outputs

    def training_step_end(self, outputs):
        return training_step_end(self, outputs)

    def validation_step_end(self, outputs):
        if outputs.get("vizualize_gt", False):
            vizualize_ground_truth(self.logger.experiment, outputs)
        if outputs.get("compute_stats", False):
            validation_step_end(self, outputs)
        if outputs.get("vizualize", False):
            step_vizualize(
                self.logger.experiment,
                self.logger.log_table,
                outputs,
                "val",
            )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        num_positive = batch["num_pos_cuts"]
        assert num_positive == 0, str(num_positive)
        outputs = self._step(
            batch,
            add_predictions_and_gt=True,
            add_embedding=self.add_embedding,
            avg_listener=True,
        )
        outputs["cuts"] = batch["cuts"]
        outputs["num_clean_cuts"] = batch["num_clean_cuts"]
        outputs["num_noisy_cuts"] = batch["num_noisy_cuts"]
        outputs["num_pos_cuts"] = batch["num_pos_cuts"]
        outputs["num_cuts"] = batch["num_cuts"]
        return outputs

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        outputs = self._step(batch, add_predictions_and_gt=True, avg_listener=True)
        outputs["num_clean_cuts"] = batch["num_clean_cuts"]
        outputs["num_noisy_cuts"] = batch["num_noisy_cuts"]
        outputs["num_pos_cuts"] = batch["num_pos_cuts"]
        outputs["num_cuts"] = batch["num_cuts"]
        if dataloader_idx == 0 or dataloader_idx is None:
            outputs["compute_stats"] = True
            return outputs
        else:
            outputs["cuts"] = batch["cuts"]
            merge_batch_and_outputs = False
            if self.vizualize_ground_truth:
                outputs["vizualize_gt"] = True
                merge_batch_and_outputs = True
                if (
                    self.current_epoch > 0
                    and self.trainer.state.stage == RunningStage.VALIDATING
                ):
                    # RunningStage could be also sanity check - we do not stop vizualizing
                    # iterate over non changing ground truth data only once
                    self.vizualize_ground_truth = False

            # Treating all dataloaders except the first one as the one for logging
            # on FEW samples and NOT COMPUTING VAL DATASET stats
            if (
                self.vizualize_every_epoch > 0
                and (self.current_epoch % self.vizualize_every_epoch) == 0
            ):
                outputs["vizualize"] = True
                merge_batch_and_outputs = True

            if merge_batch_and_outputs:
                # add both inputs and outputs (outputs overwrite inputs if keys collide)
                outputs = {**batch, **outputs}
            return outputs

    @staticmethod
    def load_from_checkpoint(ckpt_path: str, strict=True, prefer_ckpt=False, **kwargs):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        return MOSModelMixin.load_from_unpickled_checkpoint(
            ckpt, strict=strict, prefer_ckpt=prefer_ckpt, **kwargs
        )

    @staticmethod
    def load_from_unpickled_checkpoint(
        ckpt: Dict["str", Any], strict=True, prefer_ckpt=False, **kwargs
    ):
        hparams = ckpt["hyper_parameters"]

        # remove the hack that we use SSL!  The only usable models are SSL ATM and I did not save the name
        model_name = ckpt.get("name", "SSL")

        logging.info(f"Loading the model as {model_name} class")
        Model = get_MOSModel_cls(model_name)

        # always use the one from the ckpt
        fairseq_model_path = hparams["fairseq_model_path"]

        hparams = {**kwargs, **hparams} if prefer_ckpt else {**hparams, **kwargs}
        hparams["fairseq_model_path"] = fairseq_model_path
        model = Model(**hparams)
        model.load_state_dict(ckpt["state_dict"], strict=strict)
        return model



class SSL(MOSModelMixin, pl.LightningModule):
    def __init__(
        self,
        fairseq_model_path,
        num_noise_labels,
        decoder_hidden_dim: int = 768,
        decoder_dropout: float = 0.0,
        decoder_num_layers: int = 0,
        decoder_init_lin: Optional[str] = None,
        dec_betas: Optional[dict] = None,
        clip_tau: float = 0.0,
        contrast_margin: float = 0.5,
        triplet_loss_margin: float = 0.05,
        triplet_mine_neg_margin: float = 0.5,
        triplet_mine_pos_margin: float = 0.1,
        triplet_emb: Optional[str] = None,
        freeze_ssl: bool = False,
        freeze_decoder: bool = False,
        mos_projection: Optional[str] = None,
        use_deducted_mos_scores: bool = False,
        num_listeners: int = 0,
        listener_emb_size: int = 128,
        **kwargs,
    ):
        if "name" in kwargs:
            name = kwargs.pop("name")
            assert name == "SSL", name
        super().__init__(name="SSL", **kwargs)

        MOSModelMixin.__init__(self, name="SSL", **kwargs)

        self.use_deducted_mos_scores = use_deducted_mos_scores
        dec_betas = ifnone(dec_betas, DEFAULT_DEC_BETAS)

        # the state dict will be overloaded again if you use load_from_checkpoint/load_from_unpickled_checkpoint.
        # The state dict is loaded after this init function.
        self.ssl_model, ssl_model_outdim = self.load_fairseq_ssl_model(
            fairseq_model_path
        )
        if freeze_ssl:
            for p in self.ssl_model.parameters():
                p.requires_grad = False

        if decoder_num_layers == 0:
            assert (
                ssl_model_outdim == decoder_hidden_dim
            ), f"{ssl_model_outdim} vs {decoder_hidden_dim}"
            assert decoder_dropout == 0.0, str(decoder_dropout)
        self.decoder = MaxAvgPoolNN(
            ssl_model_outdim,
            out_dim=decoder_hidden_dim,
            dropout=decoder_dropout,
            num_layers=decoder_num_layers,
            init_lin=decoder_init_lin,
        )
        if freeze_decoder:
            for p in self.decoder.parameters():
                p.requires_grad = False

        # num_listeners needs to larger than ONE so the embedding makes sense to train
        # WARNING: this is not necessary true for inference (You may use a single AVG embedding)
        if num_listeners > 1 and listener_emb_size > 0:
            self.listener_emb = torch.nn.Embedding(num_listeners, listener_emb_size)
            decoder_output_dim = self.decoder.output_dim + listener_emb_size
        else:
            self.listener_emb = None
            decoder_output_dim = self.decoder.output_dim

        mos_projection = ifnone(mos_projection, "clip_range")
        if mos_projection == "clip_range":
            self.decoder2final_mos = MOSScoreProjection(decoder_output_dim)
        elif mos_projection == "clip_range_noeps":
            self.decoder2final_mos = MOSScoreProjection(
                decoder_output_dim, range_epsilon=0.0
            )
        elif mos_projection == "linear":
            self.decoder2final_mos = ScoreProjection(decoder_output_dim, 1)
        else:
            raise ValueError(f"Unknown mos_projection: {mos_projection}")

        triplet_emb = ifnone(triplet_emb, "loss")
        assert triplet_emb in ["loss", "emb"], triplet_emb
        self.triplet_emb = triplet_emb

        self.decoder2final_mos_var = ScoreProjection(decoder_output_dim, 1)
        self.decoder2final_snr = ScoreProjection(decoder_output_dim, 1)
        self.decoder2final_stoi = ScoreProjection(decoder_output_dim, 1)
        self.decoder2final_mcd = ScoreProjection(decoder_output_dim, 1)
        self.decoder2final_noise_label = LabelsProjection(
            decoder_output_dim, num_noise_labels
        )

        self.triplet_mine_neg_margin = triplet_mine_neg_margin
        self.triplet_mine_pos_margin = triplet_mine_pos_margin
        self.dec_loss = CombinedFinalLoss(
            dec_betas,
            prefix="dec_",
            clip_tau=clip_tau,
            contrast_margin=contrast_margin,
            triplet_loss_margin=triplet_loss_margin,
        )

        self.save_hyperparameters()

    def forward(self, batch):

        # TODO completely ingnoreing the returnd audio_lens
        # B * T where T is num samples in audio
        audio, _ = batch[PadAudio.PAD_AUDIO], batch[PadAudio.PAD_AUDIO_LENS]
        B = audio.shape[0]

        # B*T*F; where T is ssl_model num steps and F is embedding / feature length
        # TODO padding - mask not False
        ssl_out = self.ssl_model(audio, mask=False, features_only=True)
        ssl_feat = ssl_out["x"]
        # Steps completely ignore padding of wavs which the ssl encoder took on input
        ssl_steps = torch.full((ssl_feat.shape[0],), ssl_feat.shape[1]).to(self.device)
        ssl_final_out = self.decoder(ssl_feat, ssl_steps)

        if self.listener_emb is None:
            dec_final_out = ssl_final_out
        else:
            listener_emb = self.listener_emb(batch[CollateMOS.ANNOTATOR_ID].reshape(B))
            dec_final_out = torch.cat((ssl_final_out, listener_emb), dim=1)

        final_mos = self.decoder2final_mos(dec_final_out)
        final_mos_var = self.decoder2final_mos_var(dec_final_out)
        # https://github.com/Kyushik/Predictive-Uncertainty-Estimation-using-Deep-Ensemble/blob/master/Ensemble_Regression_ToyData_Torch.ipynb
        # Cell 7
        # Or Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles: Footnote 2 page 3
        # + 1e-6 is also added in GaussLoss in torch implementation
        final_mos_var = torch.log(1 + torch.exp(final_mos_var))
        final_snr = self.decoder2final_snr(dec_final_out)
        final_stoi = self.decoder2final_stoi(dec_final_out)
        final_mcd = self.decoder2final_mcd(dec_final_out)
        final_noise_label = self.decoder2final_noise_label(dec_final_out).reshape(B, -1)
        return (
            final_mos,
            final_mos_var,
            final_snr,
            final_stoi,
            final_mcd,
            final_noise_label,
            dec_final_out,
        )

    def _step(
        self,
        batch,
        add_predictions_and_gt=False,
        forward_outputs=None,
        add_embedding=False,
        avg_listener=False,
    ):

        if avg_listener:
            # Masked the info about the listener e.g. for inference
            AID = CollateMOS.ANNOTATOR_ID
            # Collate.ANNOTATOR_MEAN - avg listener has idx zero
            batch[AID] = torch.zeros(batch[AID].shape, dtype=torch.int).to(self.device)

        (
            final_mos,
            final_mos_var,
            final_snr,
            final_stoi,
            final_mcd,
            final_noise_label,
            dec_final_out,
        ) = self(batch)

        final_mos_mask = batch[CollateMOS.MOS_FINAL_MASK]
        # mos_deducted_mask indicates which  MOS scores were deducted from augmentation
        # eg. noise currupted speech is assinged 1.0 or human speech is assigned 5.0
        mos_deducted_mask = batch[CollateMOS.MOS_DEDUCTED_MASK]
        final_mos_true = batch[CollateMOS.MOS_FINAL]
        final_score_true = batch[
            CollateMOS.SCORE_FINAL
        ]  # Could be MOS if mean strategy or listener score

        # HACK change head for annotator agreement prediction to model confidence prediction
        # Do not need the annotator groudn truth data
        # TODO refactor
        # final_mos_var_true = batch[CollateMOS.MOS_VAR_FINAL]

        n_clean = batch["num_clean_cuts"]
        n_noise = batch["num_noisy_cuts"]
        n_pos = batch["num_pos_cuts"]
        assert (
            n_clean == n_noise or n_noise == 0
        ), f"STOI etc expect pairs but {n_clean} vs {n_noise}"

        # keep final_mos for regression
        # Below is triplet preparation
        stats = {}
        stats["num_cuts"] = float(batch["num_cuts"])
        if self.dec_loss.beta_consist_mos == 0.0:
            triplets = []
        else:
            a_p_n, npos_pairs, nneg_pairs = mine_triplets(
                final_score_true,
                self.triplet_mine_pos_margin,
                self.triplet_mine_neg_margin,
                n_clean,
                n_noise,
                n_pos,
            )
            stats["trip/n_triplets"] = float(a_p_n[0].shape[0])
            stats["trip/pos_pairs"] = float(npos_pairs)
            stats["trip/neg_pairs"] = float(nneg_pairs)

            emb = final_mos if self.triplet_emb == "loss" else dec_final_out
            triplets = [
                tuple(index_select(emb, 0, idx.to(self.device)) for idx in a_p_n)
            ]

        if self.use_deducted_mos_scores:
            contrastive_loss_mask = torch.logical_or(mos_deducted_mask, final_mos_mask)
        else:
            contrastive_loss_mask = final_mos_mask

        dec_loss, dec_stats = self.dec_loss(
            self.device,
            n_clean,
            final_mos_mask,
            contrastive_loss_mask,
            final_mos,
            final_score_true,
            # WARNING HACK
            # changed the purpose of final_mos_var from predicting annotator variance
            # to predict model confidence
            final_mos_var,
            # ORDER IS SUPER IMPORTANT - betas - weights are applied based on it
            # SNR, STOI does not need to be masked - always correct value
            [
                # Predicting SNR which we know from the data augmentation
                (final_snr, batch[CollateNoiseLabels.SNR]),
                # discarding predictions for positive
                (final_stoi[: n_clean + n_noise], batch[CollateSTOI_MCD.STOI]),
                (final_mcd[: n_clean + n_noise], batch[CollateMCD.MCD]),
            ],
            final_noise_label,
            batch[CollateNoiseLabels.LABEL],
            triplets,
        )

        stats = {**stats, **dec_stats}
        outputs = {"loss": dec_loss, "stats": stats}

        if add_predictions_and_gt:
            # Adding helper variables to the output dict (for logging and stats)
            # The below does not work for inference since the final_mos_mask will mask it out
            # since it depends on ground truth
            # outputs[f"pred_{CollateMOS.MOS_FINAL}"] = final_mos_regress.detach()
            # outputs[f"true_{CollateMOS.MOS_FINAL}"] = final_mos_true_regress[:n_clean].detach()
            # The positive examples depends on number of clean cuts so it works also for inference
            outputs[f"pred_{CollateMOS.MOS_FINAL}"] = final_mos[:n_clean].detach()
            outputs[f"true_{CollateMOS.SCORE_FINAL}"] = final_score_true[
                :n_clean
            ].detach()
            outputs[f"pred_{CollateMOS.MOS_VAR_FINAL}"] = final_mos_var[
                :n_clean
            ].detach()
            # outputs[f"true_{CollateMOS.MOS_VAR_FINAL}"] = final_mos_var_true[:n_clean].detach()
            outputs[f"true_{CollateMOS.MOS_FINAL}"] = final_mos_true[:n_clean].detach()
            outputs[f"predmix_{CollateMOS.MOS_FINAL}"] = final_mos[
                : n_clean + n_noise
            ].detach()
            outputs[f"predmix_{CollateMOS.MOS_VAR_FINAL}"] = final_mos_var[
                : n_clean + n_noise
            ].detach()
        if add_embedding:
            outputs["dec_final_out"] = dec_final_out.detach()
        return outputs

    @staticmethod
    def load_fairseq_ssl_model(model_path):
        import fairseq

        ssl_model_type = model_path.split("/")[-1]
        if ssl_model_type == "wav2vec_small.pt":
            ssl_out_dim = 768
        elif ssl_model_type in ["w2v_large_lv_fsh_swbd_cv.pt", "xlsr_53_56k.pt"]:
            ssl_out_dim = 1024
        else:
            raise ValueError(f"Unknown ssl_model_type {ssl_model_type}")

        try:
            model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [model_path]
            )
        except Exception as e:
            if ssl_model_type in ["w2v_large_lv_fsh_swbd_cv.pt", "xlsr_53_56k.pt"]:
                msg = (
                    "See the comment on the line before the exceptions oplatek fixed"
                    "the problem by copying these fields below "
                    "to fairseq/tasks/audio_pretraining.py:AudioPretrainingConfig"
                )
            else:
                msg = ""
            logging.exception(msg)
            raise e
            # # Seq2Seq models during fine-tuning
            # eval_wer: bool = field(
            #     default=False, metadata={"help": "compute WER for Seq2Seq models"}
            # )
            # eval_wer_config: GenerationConfig = field(
            #     default_factory=lambda: GenerationConfig(),
            #     metadata={"help": "beam search config for evaluating wer during training"},
            # )
            # eval_wer_tokenizer: Any = field(
            #     default=None,
            #     metadata={"help": "tokenizer config for evaluating wer during training"},
            # )
            # eval_wer_post_process: str = field(
            #     default="letter",
            #     metadata={
            #         "help": "remove BPE tokens before scoring (can be sentencepiece, letter, and more)"
            #     },
            # )
            # autoregressive: bool = field(
            #     default=False,
            #     metadata={
            #         "help": "required for autoregressive decoders (like seq2seq models); "
            #         "adds 'prev_output_tokens' to input and appends eos to target"
            #     },
            # )

        ssl_model = model[0]
        ssl_model.remove_pretraining_modules()
        return ssl_model, ssl_out_dim
