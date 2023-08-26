import pytest
import numpy as np
from moosenet.plda_utils import calc_bins_edges


def test_calc_bins_edges_keep_same_large_nbins():
    labels = np.arange(1.0, 5.0, 0.1)  # All labels are unique
    n_labels = len(labels)
    for n_bins in range(n_labels, n_labels + 10):
        bin_edges, bins_counts = calc_bins_edges(labels, n_bins, min_bin_count=1)
        assert (bins_counts == np.ones(labels.shape)).all()
        assert len(bin_edges) + 1 == n_labels


def test_calc_bin_edges_reduce_nbins():
    labels = np.arange(1.0, 5.0, 0.1)  # All labels are unique
    n_labels = len(labels)
    for n_bins in range(n_labels - 10, n_labels):
        bin_edges, bins_counts = calc_bins_edges(labels, n_bins, min_bin_count=1)
        assert len(bin_edges) + 1 == n_bins


def test_calc_bin_edges_raises_when_too_few_labels():
    labels = np.arange(1.0, 5.0, 0.1)
    with pytest.raises(AssertionError):
        calc_bins_edges(np.array([0, 1]), 10, min_bin_count=3)
    with pytest.raises(AssertionError):
        calc_bins_edges(labels, 10, min_bin_count=len(labels) + 1)


def test_calc_bin_edges_merges_so_bins_are_large_enough():
    labels = np.arange(1.0, 5.0, 0.1)
    mbc = 2
    bin_edges, bins_counts = calc_bins_edges(labels, 10, min_bin_count=2)
    assert (np.array(bins_counts) >= mbc).all(), str(bins_counts)

    bin_edges, bins_counts = calc_bins_edges(labels, len(labels), min_bin_count=2)
    assert (np.array(bins_counts) >= mbc).all(), str(bins_counts)


def test_calc_bin_edges_do_not_collapse():
    labels = np.array([2.125, 4.25, 3.75, 1.375, 3.5])
    bin_edges, bins_counts = calc_bins_edges(labels, 20, min_bin_count=2)
    assert bins_counts == [2, 3], bins_counts
