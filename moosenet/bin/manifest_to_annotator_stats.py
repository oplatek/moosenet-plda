from lhotse import CutSet
import csv
import click
from statistics import variance, mean
from collections import defaultdict
from moosenet.datasetup import load_cuts
from moosenet.bin import moosenet


def load_utt_spk_mos_stats(cuts: CutSet):
    for c in cuts:
        uttid = c.id
        mosd = c.supervisions[0].custom["MOS"]
        for annotatorid, mos in mosd.items():
            yield uttid, annotatorid, mos


def compute_utt_mean(t):
    dsum = defaultdict(float)
    dcount = defaultdict(float)
    for uttid, _, mos in t:
        dsum[uttid] += mos
        dcount[uttid] += 1
    return dict((k, dsum[k] / dcount[k]) for k in dsum)


def compute_annotator_mean_variance(t):
    tmp = defaultdict(list)
    for _, aid, mos in t:
        tmp[aid].append(mos)

    meand = dict((aid, mean(moss)) for aid, moss in tmp.items())
    vard = dict((aid, variance(moss, meand[aid])) for aid, moss in tmp.items())
    return meand, vard


def compute_annotator_correction(utt_mos_mean, annot_mean, t):
    aid2utts = defaultdict(list)
    for uttid, aid, mos in t:
        aid2utts[aid].append(uttid)

    annot_correction = {}
    for aid, utts in aid2utts.items():
        # mean of means(subsets of X)  is not mean of X ;
        # but really good approximation for our data (because of all annotators have 4 annotations)
        annotor_judged_utt_mean = mean(utt_mos_mean[u] for u in utts)
        annot_avg = annot_mean[aid]
        annot_correction[aid] = annotor_judged_utt_mean - annot_avg

    return annot_correction


@moosenet.command()
@click.argument("manifest_paths", nargs=-1)
@click.argument("outcsv", type=click.Path(exists=False))
def manifest_to_annototar_stats(manifest_paths, outcsv):
    cuts = load_cuts(*manifest_paths)
    t = list(load_utt_spk_mos_stats(cuts))
    utt_mos_mean = compute_utt_mean(t)
    annot_mean, annotator_variance = compute_annotator_mean_variance(t)
    annotator_correction = compute_annotator_correction(utt_mos_mean, annot_mean, t)

    with open(outcsv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(
            (aid, annotator_correction[aid], annotator_variance[aid])
            for aid in annotator_correction.keys()
        )
