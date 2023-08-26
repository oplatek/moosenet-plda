import torch
from lhotse.utils import ifnone
import logging
import math
from torch import Tensor, BoolTensor
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
import torch.optim.lr_scheduler
from typing import Tuple, List, Dict, Optional


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive Loss
    Args:
        margin: non-neg value, the smaller the stricter the loss will be, default: 0.2

    See UTMOS: https://arxiv.org/pdf/2204.02152.pdf
    """

    def __init__(self, margin=0.2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, pred_score, gt_score):
        if pred_score.dim() > 2:
            pred_score = pred_score.mean(dim=1).squeeze(1)
        # pred_score, gt_score: tensor, [batch_size]
        gt_diff = gt_score.unsqueeze(1) - gt_score.unsqueeze(0)
        pred_diff = pred_score.unsqueeze(1) - pred_score.unsqueeze(0)
        loss = torch.maximum(
            torch.zeros(gt_diff.shape).to(gt_diff.device),
            torch.abs(pred_diff - gt_diff) - self.margin,
        )
        loss = loss.mean().div(2)
        return loss


# so we do not normalize with masked out cuts
def logCoshLoss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """1 / N * sum_{i=1}^{N} ln(cosh(x_i)) where x_i = y_i - yhat_i
    Numericly stable version based on:
        https://datascience.stackexchange.com/questions/96271/logcoshloss-on-pytorch
        https://pytorch.org/docs/stable/generated/torch.nn.functional.softplus.html#torch.nn.functional.softplus
    Credit:
        MIT license
        https://github.com/tuantle/regression-losses-pytorch
    """
    x = y_pred - y_true
    # logCosh
    loss = x + F.softplus(-2.0 * x) - math.log(2.0)
    # loss keeps the x shape
    return loss


class LogCoshLoss(torch.nn.Module):
    def __init__(self, clip_tau=0.0):
        super().__init__()
        assert 0 <= clip_tau, clip_tau
        self.tau = torch.tensor(clip_tau, dtype=torch.float)

    def forward(self, y_pred, y_true, mask: Optional[BoolTensor] = None):
        if mask is not None:
            y_pred[~mask] = 0.0
            y_true[~mask] = 0.0
        loss = logCoshLoss(y_pred, y_true)
        if self.tau > 0.0:
            noise_ignore_mask = torch.abs(y_pred - y_true) > self.tau
            loss[~noise_ignore_mask] = 0.0
        loss = torch.mean(loss)
        return loss


def mine_triplets(
    mos_scores_gt: Tensor,
    positive_margin,
    negative_margin,
    num_clean,
    num_noisy,
    num_positive,
):
    """Return indexes to embeddings for [(anchor, positive, negative),...] triplets

    We assume that K == mos_scores.shape[0] and K == embeddings.shape[0]
    and the indexes are selected from [0, ..., K)

    We further assume there are first MOS scores for original cuts than for noise augmented cuts
    and finally positive cuts (typical just one) which are original cuts with different MOS preserving augmentation.

    Note for usage: Embeddings could be predicted MOS score (after projection) or embedding before projection
    """

    assert positive_margin > 0.0
    assert positive_margin < negative_margin
    assert (
        negative_margin <= 0.5
    ), "embeddings for MOS scores of different int values should be far from each other"

    # compute distances so we can select anchor / positive pairs
    # Note: we are sure that there will be at least one pair
    # we augmented a single cut with MOS preserving augmentations twice

    # assing the noisy cuts insane MOS scores so they will be distant from everything else
    mos = mos_scores_gt
    total_cuts = num_clean + num_noisy + num_positive
    assert mos.shape == (total_cuts, 1), (total_cuts, mos.shape)
    mos_2d_diff = torch.abs(mos.unsqueeze(1) - mos.unsqueeze(0))

    clean_and_positive_idx = list(range(num_clean)) + list(
        range(total_cuts - num_positive, total_cuts)
    )

    # Indexes of triplets
    triplets = []  # [(a0, p0, n0), (a0, p0, n1), ..., (a0, p1, n0), ...]

    # anchor must be from clean cuts
    pos_pairs, neg_pairs = 0, 0
    for a in range(num_clean):
        for p in clean_and_positive_idx:
            # positive candidate to anchor can be any cut from clean or positive cuts close enough
            # INCLUDING the cut itself
            if mos_2d_diff[p, a] >= positive_margin:
                continue
            # p != a and mos_2d_diff[p, a] < positive_margin
            pos_pairs += 1

            # one negative candidate is the noisy alternative of the anchor
            triplets.append((a, p, a + num_clean))
            neg_pairs += 1

            # or any other candidate clean or positive far enough
            for n in clean_and_positive_idx:
                if mos_2d_diff[a, n] < negative_margin:
                    continue
                triplets.append((a, p, n))
                neg_pairs += 1
                if n < num_clean and mos[n] < mos[a]:
                    # clean examples have also noisy variant and
                    # if they are worse than anchor noisy variant is even worse
                    triplets.append((a, p, n))
                    neg_pairs += 1

    if len(triplets) == 0:
        anchor, positive, negative = [], [], []
        logging.warning(
            f"No triplets mined - no positive cut prepare for cuts with MOS scores {mos}"
        )
    else:
        anchor, positive, negative = zip(*triplets)

    anchor = torch.IntTensor(anchor)
    positive = torch.IntTensor(positive)
    negative = torch.IntTensor(negative)
    return (anchor, positive, negative), pos_pairs, neg_pairs


class CombinedFinalLoss(torch.nn.Module):
    def __init__(
        self,
        betas: Dict[str, float],
        prefix=None,
        triplet_loss_margin=0.1,
        contrast_margin=0.2,
        clip_tau=0.0,
    ):
        super().__init__()
        assert all([v >= 0.0 for v in betas.values()]), str(betas)
        self.prefix = ifnone(prefix, "")
        logging.info(f"Losses {self.prefix} Dict: {betas}")

        self.betas = betas
        self.final_mos = LogCoshLoss(clip_tau=clip_tau)
        self.log_cosh = LogCoshLoss(clip_tau=0.0)

        self.noise_label_CE = torch.nn.CrossEntropyLoss()

        # TODO implement lossless triplet loss and compare
        #        # http://conf.uni-obuda.hu/sami2021/89_SAMI.pdf
        #        # Margin if not using meanOS should be 0.99
        # L(a, p, n) = max(dist(a, p) - d(a, n) + margin, 0}
        #  easy triplets have loss 0 because d(a, p) + margin < d(a, n)
        #  hard triplets d(a, n) < d(a, p) - negative is closer to anchor than positive
        #  semi-hard triples d(a, p) < d(a, p) as wanted but still not far enough! push it away
        #
        # For swap=True see http://www.bmva.org/bmvc/2016/papers/paper119/paper119.pdf
        # We do the swap ourselves - if we mine (a, p, X) then we mine also (p, a, X)
        #
        # Note we use use the clip logCoshLoss as we would use for regression
        self.trip_consist_mos = torch.nn.TripletMarginWithDistanceLoss(
            margin=triplet_loss_margin,
            swap=False,
            distance_function=self.final_mos,
        )
        self.contrast_mos = ContrastiveLoss(contrast_margin)

        self.gaussloss = torch.nn.GaussianNLLLoss()

        self.beta_mos_final = betas["mos_final"]
        self.beta_noise_label = betas["noise_label"]
        self.beta_consist_mos = betas["consist_mos"]
        self.beta_contrast_mos = betas["contrast_mos"]
        self.beta_var_mos_final = betas["var_mos_final"]

        # NOTE order is important and should match DEFAULT_DEC_BETAS
        betas_regress_keys = ["snr", "stoi", "mcd"]
        self.betas_regress = dict([(k, betas[k]) for k in betas_regress_keys])

    def forward(
        self,
        device,
        n_clean,
        mos_mask,
        contrast_mask,
        mos_pred,
        mos_gtrue,
        pred_var_mos,
        regression_metrics,  # List[Tuple(torch.Tensor, torch.Tensor)]
        pred_noise_label,
        true_noise_label,
        triplet_metrics,  # List[Tuple(torch.Tensor, torch.Tensor, torch.Tensor)]
    ):
        assert len(regression_metrics) == len(
            self.betas_regress
        ), f"{len(regression_metrics)} vs {len(self.betas_regress)}: betas {self.betas_regress}"
        assert (
            self.beta_consist_mos == 0 or len(triplet_metrics) == 1
        ), "We support only consist_mos metric ATM"

        stats = {}
        loss = torch.tensor(0.0).to(device)
        P = self.prefix

        if self.beta_mos_final > 0:
            mos_pred_final = mos_pred.clone()
            mos_true_final = mos_gtrue.clone()
            mos_pred_final[~mos_mask] = 0.0
            mos_true_final[~mos_mask] = 0.0
            mos = self.final_mos(mos_pred_final, mos_true_final)
            stats[f"{P}mos"] = mos.detach()
            loss += self.beta_mos_final * mos

        if self.beta_noise_label > 0.0:
            true_noise_label = true_noise_label.reshape(-1)

            ce_noise_loss = self.noise_label_CE(pred_noise_label, true_noise_label)
            stats[f"{P}noise_label"] = ce_noise_loss.detach()
            loss += self.beta_noise_label * ce_noise_loss

        if self.beta_consist_mos > 0.0:
            # TODO support more triplet metrics
            # anchor, positive, negative
            a, p, n = triplet_metrics[0]
            trip_consist_mos = self.trip_consist_mos(a, p, n)
            stats[f"{P}consist_mos"] = trip_consist_mos.detach()
            loss += self.beta_consist_mos * trip_consist_mos

        # relying heavily that dict keeps order (Python3.7+)
        for (name, beta), (pred, true) in zip(
            self.betas_regress.items(), regression_metrics
        ):
            if beta <= 0.0:
                continue
            reg_loss = self.log_cosh(pred, true)
            loss += beta * reg_loss
            stats[f"{P}{name}"] = reg_loss.detach()

        if self.beta_contrast_mos > 0.0:
            score_pred_contr = mos_pred.clone()
            score_true_contr = mos_gtrue.clone()
            score_pred_contr[~contrast_mask] = 0.0
            score_true_contr[~contrast_mask] = 0.0
            contr_loss = self.contrast_mos(score_pred_contr, score_true_contr)
            loss += self.beta_contrast_mos * contr_loss
            stats[f"{P}contrast_mos"] = contr_loss.detach()

        if self.beta_var_mos_final:
            mos_pred_forvar = mos_pred.clone()[:n_clean]
            mos_true_forvar = mos_gtrue.clone()[:n_clean]
            pred_var_mos = pred_var_mos[:n_clean]
            gauss_loss = self.gaussloss(mos_pred_forvar, mos_true_forvar, pred_var_mos)
            loss += self.beta_var_mos_final * gauss_loss
            stats[f"{P}gauss_loss"] = gauss_loss

        return loss, stats


class Regress_CTC_NoiseCE_Loss(torch.nn.Module):
    def __init__(self, betas: Dict[str, float], prefix=None):
        super().__init__()
        raise NotImplementedError("Deprecated need to update")
        self.prefix = ifnone(prefix, "")
        logging.info(f"Losses {self.prefix} Dict: {betas}")

        # make sure the blank idx is reserved for blanks when used
        self.ctc_loss = torch.nn.CTCLoss(blank=0)

        self.log_cosh = LogCoshLoss(clip_tau=0.0)

        self.noise_label_CE = torch.nn.CrossEntropyLoss()
        self.set_betas(betas)

    def set_betas(self, betas):
        assert all([v >= 0.0 for v in betas.values()]), str(betas)
        self.ctc_phntrn_beta = (
            betas.pop("ctc_phntrn") if "ctc_phntrn" in betas else betas
        )  # mandatory item
        self.beta_noise_label = (
            betas.pop("noise_label") if "noise_label" in betas else betas
        )
        self.betas = betas

    def compute_ctc_loss(
        self, ctc_probs, ctc_tokens, ctc_probs_lens, ctc_token_lens, device
    ):
        # If not used avoid computing it and potential NaN and speed improvement
        empty_batch = ctc_tokens is None or torch.sum(ctc_token_lens) == 0
        if empty_batch:
            return torch.tensor(0.0).to(device)

        # account for subsampling and for different audio length between ref and tts audio
        # C is the smallest possible so CTC can work
        C = torch.max(torch.ceil(ctc_token_lens / ctc_probs_lens)).to(torch.int)
        # (B, T, ntokens) -> (B, C * T, ntokens)
        ctc_probs = torch.repeat_interleave(ctc_probs, C, dim=1)
        ctc_probs_lens = C * ctc_probs_lens

        # (B, T, ntokens) -> (T, B, ntokens) for CTCLoss
        ctc_probs = ctc_probs.permute(1, 0, 2)
        ctc_loss = self.ctc_loss(ctc_probs, ctc_tokens, ctc_probs_lens, ctc_token_lens)
        return ctc_loss

    def forward(
        self,
        device,
        regression_metrics,  # List[Tuple(torch.Tensor, torch.Tensor)]
        pred_noise_label,
        true_noise_label,
        ctc_probs,
        ctc_tokens,
        ctc_probs_lens,
        ctc_token_lens,
    ):
        assert len(regression_metrics) == len(
            self.betas
        ), f"{len(regression_metrics)} vs {len(self.betas)}: betas {self.betas}"

        stats = {}
        loss = torch.tensor(0.0).to(device)
        P = self.prefix

        if self.beta_noise_label > 0.0:
            T = pred_noise_label.shape[1]
            # (B, T, num_labels) -> (B * T, num_labels)
            # (B, num_labels)
            pred_noise_label = pred_noise_label.reshape(-1, pred_noise_label.shape[-1])
            true_noise_label = true_noise_label.repeat((T, 1)).reshape(-1)
            ce_noise_loss = self.noise_label_CE(pred_noise_label, true_noise_label)
            stats[f"{P}noise_label"] = ce_noise_loss.detach()
            loss += self.beta_noise_label * ce_noise_loss

        if self.ctc_phntrn_beta > 0.0:
            ctc_phntrn_loss = self.compute_ctc_loss(
                ctc_probs,
                ctc_tokens,
                ctc_probs_lens,
                ctc_token_lens,
                device,
            )
            stats[f"{P}ctc_phntrn"] = ctc_phntrn_loss.detach()
            loss += self.ctc_phntrn_beta * ctc_phntrn_loss

        # relying heavily that dict keeps order (Python3.7+)
        for (name, beta), (pred, true) in zip(self.betas.items(), regression_metrics):
            if beta <= 0.0:
                continue
            reg_loss = self.log_cosh(pred, true)
            loss += beta * reg_loss
            stats[f"{P}{name}"] = reg_loss.detach()

        return loss, stats
