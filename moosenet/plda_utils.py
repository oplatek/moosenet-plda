import plda
import logging
import torch
import torch.nn.functional as F
from numpy import digitize
import numpy as np
from copy import deepcopy
from numpy import arange
from moosenet.collate import CollateMOS

MODEL_CONFIDENCE = "model_confidence"


def calc_bins_edges(labels, n_bins, min_bin_count=2):
    # digitize automatically classifies the edges cases as 0 bin and K bin
    # plus we need all the data fit there even if the bins_edges[0] > 1 or bins_edges[K-1] < 5
    assert n_bins > 0
    unique, counts = np.unique(labels, return_counts=True)
    assert (
        np.sum(counts) >= min_bin_count
    ), f"At least one bin with {min_bin_count} needs to be created"

    # Sorted bins_centers with their counts
    bins_centers = dict(sorted(zip(unique, counts)))
    bck = list(bins_centers.items())
    # edges with their count from [edge between last two centers , edge between last two centers]
    bins_edges = [(a[0] + b[0]) / 2.0 for a, b in zip(bck[:-1], bck[1:])]
    # first count is for bin on the left of first edge ie count of the first center,
    # the rest is calculated as count of the center bin on the right from all edges including the first.
    bins_counts = [bck[0][1]] + [b_count[1] for b_count in bck[1:]]
    merge_bin_counts = [a + b for a, b in zip(bins_counts[:-1], bins_counts[1:])]
    min_merge_bin_counts = min(merge_bin_counts)
    mbc = min(bins_counts)  # TODO O(n)

    if len(bins_counts) < n_bins:
        logging.warning(
            f"You requested n_bins {n_bins} but there is only {len(bins_counts)} classes"
        )
    n_bins = min(n_bins, len(bins_counts))
    logging.info(f"Using {n_bins=}")

    assert len(bins_edges) == len(
        merge_bin_counts
    ), "should hold always for algorithm below"
    # merging two neighbourghing bins lowest count
    while len(merge_bin_counts) + 1 > n_bins or mbc < min_bin_count:
        for i, merge_bin_count in enumerate(merge_bin_counts):
            if merge_bin_count == min_merge_bin_counts:
                break
        # deleting an edge
        bins_edges.pop(i)
        # neighbouring merge_bin_counts needs to be updated
        if i > 0:
            merge_bin_counts[i - 1] += bins_counts[i]
        if i < len(merge_bin_counts) - 1:
            merge_bin_counts[i + 1] += bins_counts[i]
        merge_bin_counts.pop(i)  # TODO O(n)
        # bins_counts needs to be updated after merge_bins_counts
        bins_counts[i + 1] += bins_counts[i]
        bins_counts.pop(i)  # TODO O(n)
        if len(merge_bin_counts) == 0:
            break  # we need to keep at least 1 bin
        else:
            min_merge_bin_counts = min(merge_bin_counts)  # TODO O(n)
            mbc = min(bins_counts)  # TODO O(n)

    sbc = sum(bins_counts)
    sbcv = sum(bins_centers.values())
    assert sbc == sbcv, f"Merging bins maintains counts {sbc} vs {sbcv}"
    # logging.info(f"
    return bins_edges, bins_counts


def fit_plda(batches, n_bins=32, add_gauss_noise=0.0, multiply_data=1, **fit_kwargs):
    # 32 n_bins 8 judges * (5 - 1) discrite human scores

    if multiply_data > 1.0:
        if add_gauss_noise == 0.0:
            logging.warning("Multiplying data without noise does not make sense")
        batches = batches * multiply_data
        # embeddings = np.tile(embeddings, (multiply_data, 1))
    embeddings = np.vstack([extract_embeddings(b) for b in batches])

    if add_gauss_noise > 0.0:
        embeddings = embeddings + np.random.normal(
            0, np.var(embeddings) * add_gauss_noise, embeddings.shape
        )
    labels = np.vstack([extract_labels(b) for b in batches]).squeeze()

    n_samples = embeddings.shape[0]
    n_feats = embeddings.shape[1]
    logging.info(f"plda fitting with {n_samples=}, {n_feats=}, {fit_kwargs=}")

    bins_edges, _bins_counts = calc_bins_edges(labels, n_bins)
    labels = np.digitize(labels, bins_edges)
    plda_classifier = plda.Classifier()
    plda_classifier.fit_model(embeddings, labels, **fit_kwargs)
    return plda_classifier, bins_edges


def weighted_bins(bins_centers, logpps_k):
    probs = np.exp(logpps_k)
    B = probs.shape[0]
    prob_weighted_mos = np.sum(probs * bins_centers, axis=1).reshape(B, 1)
    return prob_weighted_mos


def argmax_bins(bins_centers, logpps_k):
    B = logpps_k.shape[0]
    greedy_mos = np.array(
        [bins_centers[i] for i in np.argmax(logpps_k, axis=-1).tolist()]
    ).reshape(B, 1)
    return greedy_mos


def create_kernel(num_elements, sigma=0.6, zero_threshold=0.001):
    # inspired by https://matthew-brett.github.io/teaching/smoothing_as_convolution.html
    num_elements >= 2.0
    assert zero_threshold > 0
    assert sigma > zero_threshold
    x = list(range(1, int((num_elements + 1) // 2)))
    x_for_kernel = np.array(list(reversed([-i for i in x])) + [0] + x)
    kernel = np.exp(-((x_for_kernel) ** 2) / (2 * sigma**2))
    kernel_above_thresh = kernel > zero_threshold
    finite_kernel = kernel[kernel_above_thresh]
    finite_kernel = finite_kernel / finite_kernel.sum()
    if finite_kernel.size < num_elements - 1:
        logging.warning(
            f"REQUESTED kernel size {num_elements} but for sigma {sigma}"
            f" there are ONLY {finite_kernel.size} zero elements"
        )
    return finite_kernel


def infer_pldaXxY(plda_classifier, bins_edges, batches, weight_bins=True):
    batches = deepcopy(batches)

    ext_bins_edges = [1.0] + bins_edges + [5.0]
    bins_centers = [
        (a + b) / 2.0 for a, b in zip(ext_bins_edges[:-1], ext_bins_edges[1:])
    ]
    assert len(bins_centers) + 1 == len(ext_bins_edges), (
        len(bins_centers),
        len(ext_bins_edges),
    )
    bins2mos = weighted_bins if weight_bins else argmax_bins
    print(bins2mos)

    # TODDO
    #     U_model = better_classifier.model.transform(training_data, from_space='D', to_space='U_model')
    #     U_datum_0 = U_model[0][None,]
    #     U_datum_1 = U_model[1][None,]
    #
    #     log_ratio_0_1 = better_classifier.model.calc_same_diff_log_likelihood_ratio(U_datum_0, U_datum_1)
    #
    #     print(log_ratio_0_1)
    #
    # TODOO

    for batch in batches:
        embeddings = extract_embeddings(batch)
        _pred_labels, logpps_k = plda_classifier.predict(
            embeddings, normalize_logps=True
        )
        if len(logpps_k.shape) == 1:
            logpps_k = logpps_k.reshape(1, logpps_k.shape[0])

        batch[f"pred_{CollateMOS.MOS_FINAL}"] = 88.88
        batch[MODEL_CONFIDENCE] = 88.88
    return batches


def infer_plda(
    plda_classifier,
    bins_edges,
    batches,
    weight_bins=True,
    kernel_size=10,
    kernel_sigma=2.0,
):
    batches = deepcopy(batches)

    ext_bins_edges = [1.0] + bins_edges + [5.0]
    bins_centers = [
        (a + b) / 2.0 for a, b in zip(ext_bins_edges[:-1], ext_bins_edges[1:])
    ]
    assert len(bins_centers) + 1 == len(ext_bins_edges), (
        len(bins_centers),
        len(ext_bins_edges),
    )
    bins2mos = weighted_bins if weight_bins else argmax_bins

    kernel = create_kernel(kernel_size, sigma=kernel_sigma)
    logging.info(f"Using smoothing kernel {kernel}")
    kernel_half = kernel.size // 2
    assert (2 * kernel_half) + 1 == kernel.size

    for batch in batches:
        embeddings = extract_embeddings(batch)
        _pred_labels, logpps_k = plda_classifier.predict(
            embeddings, normalize_logps=True
        )
        if len(logpps_k.shape) == 1:
            logpps_k = logpps_k.reshape(1, logpps_k.shape[0])
        pred_mos = bins2mos(bins_centers, logpps_k)

        pred_mos_idx = np.digitize(pred_mos.squeeze(), bins_edges)

        # single bin posterior does not work great
        # pred_mos_posterior = logpps_k[np.arange(logpps_k.shape[0]), pred_mos_idx]

        pred_mos_1hot = F.one_hot(
            torch.LongTensor(pred_mos_idx), logpps_k.shape[1]
        ).float()
        # TODO make it batched operation
        for i in range(pred_mos_1hot.shape[0]):
            smooth_1hot = np.convolve(pred_mos_1hot[i, :], kernel)
            if kernel_half > 0:
                # keep the original size
                smooth_1hot = smooth_1hot[kernel_half:-kernel_half]
            # normalize back to sum to one (because of the edges of convolved array)
            smooth_1hot /= np.sum(smooth_1hot)
            pred_mos_1hot[i, :] = torch.FloatTensor(smooth_1hot)

        # KL divergence is too complicated
        # kl = F.kl_div(torch.FloatTensor(logpps_k), pred_mos_1hot, reduction="none").sum(dim=1)

        smooth_prob = (
            np.sum(np.exp(logpps_k) * pred_mos_1hot.numpy(), axis=1) / kernel.size
        )

        assert not np.isnan(
            pred_mos
        ).any(), f"for batch {batch} there is NaN in {pred_mos}"
        assert not np.isinf(
            pred_mos
        ).any(), f"for batch {batch} there is Inf in {pred_mos}"
        assert pred_mos.shape == batch[f"pred_{CollateMOS.MOS_FINAL}"].shape
        batch[f"pred_{CollateMOS.MOS_FINAL}"] = pred_mos
        # batch[MODEL_CONFIDENCE] = kl
        # batch[MODEL_CONFIDENCE] = pred_mos_posterior
        batch[MODEL_CONFIDENCE] = smooth_prob
    return batches


def extract_embeddings(batch):
    n_clean = batch["num_clean_cuts"]
    batched_embeddings = batch["dec_final_out"][:n_clean, ...]
    # batched_embeddings = torch.log(1 + torch.exp(batched_embeddings))
    return batched_embeddings.numpy()


def extract_labels(batch):
    n_clean = batch["num_clean_cuts"]
    batched_labels = batch[f"true_{CollateMOS.MOS_FINAL}"][:n_clean, ...].numpy()
    return batched_labels
