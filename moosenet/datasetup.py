import csv
import math
import numpy as np
import logging
from lhotse import load_manifest_lazy, CutSet, load_manifest
from lhotse.utils import Pathlike, ifnone
from lhotse.manipulation import combine
from moosenet import exp_dir, data_dir


def load_cuts(*cuts_manifests: Pathlike):
    return sum(
        [load_manifest_lazy(m) for m in cuts_manifests[1:]],
        load_manifest_lazy(cuts_manifests[0]),
    )


def get_manifest(data_setup: str, train_ratio: float = 1.0):
    assert 0.0 <= train_ratio <= 1, f"Should be 0.0 <= {train_ratio=} <= 1.0"
    if "voicemos_main1" in data_setup:
        voicemos_main1_val = load_cuts(
            data_dir / "voicemos" / "cuts_main1_dev.jsonl.gz"
        )
        voicemos_main1_tr = load_cuts(
            data_dir / "voicemos" / "cuts_main1_train.jsonl.gz"
        )
        voicemos_main1_tr = voicemos_main1_tr.filter(
            lambda c: c.id != "sys4bafa-uttc2e86f6-326-0"
        )
        voicemos_main1_test = load_cuts(
            data_dir / "voicemos" / "cuts_main1_test.jsonl.gz"
        )

    if "voicemos_ood1" in data_setup:
        voicemos_ood1_val = load_cuts(data_dir / "voicemos" / "cuts_ood1_dev.jsonl.gz")
        voicemos_ood1_tr = load_cuts(data_dir / "voicemos" / "cuts_ood1_train.jsonl.gz")
        voicemos_ood1_test = load_cuts(
            data_dir / "voicemos" / "cuts_ood1_test.jsonl.gz"
        )

    if data_setup == "vcc2018":
        val_cuts = load_cuts(data_dir / "vcc2018" / "cuts.val.jsonl.gz")
        train_cuts = load_cuts(data_dir / "vcc2018" / "cuts.train.jsonl.gz")
        val_egs_cuts = [val_cuts.subset(first=10)]
        test_cuts = None
    elif data_setup == "voicemos_main1":
        val_cuts = voicemos_main1_val
        train_cuts = voicemos_main1_tr
        val_egs_cuts = [val_cuts.subset(first=10)]
        test_cuts = voicemos_main1_test
    elif data_setup == "voicemos_ood1_labeledonly":
        val_cuts = voicemos_ood1_val
        train_cuts = voicemos_ood1_tr
        val_egs_cuts = [val_cuts.subset(first=10)]
        test_cuts = voicemos_ood1_test
    elif data_setup == "voicemos_ood1_combined_unlabeled":
        unlabled_cuts = CutSet.from_manifests(
            recordings=load_manifest(
                data_dir / "voicemos" / "recordings_ood1_unlabeled.jsonl.gz"
            )
        )
        val_cuts = voicemos_ood1_val
        train_cuts = voicemos_ood1_tr
        train_cuts = combine(unlabled_cuts, train_cuts)
        val_egs_cuts = [val_cuts.subset(first=10)]
        test_cuts = voicemos_ood1_test
    elif data_setup == "voicemos_ood1_mux_unlabeled":
        # TODO and switch to loading directly from cutset
        unlabled_cuts = CutSet.from_manifests(
            recordings=load_manifest(
                data_dir / "voicemos" / "recordings_ood1_unlabeled.jsonl.gz"
            )
        )
        val_cuts = voicemos_ood1_val
        train_cuts = voicemos_ood1_tr
        train_cuts = CutSet.mux(
            voicemos_ood1_tr,
            unlabled_cuts,
            # num cuts in cutsets so their probability is proportional and are depleted around the same time
            weights=[len(voicemos_ood1_tr), len(unlabled_cuts)],
        )
        val_egs_cuts = [val_cuts.subset(first=10)]
        test_cuts = voicemos_ood1_test
    elif data_setup == "libritts_vctk_voicemos_main1":
        val_libritts_clean = load_cuts(
            data_dir / "libritts" / "cuts_dev-clean.jsonl.gz"
        )
        val_vctk = load_cuts(data_dir / "vctk" / "cuts_dev.jsonl.gz")
        val_cuts = combine(voicemos_main1_val, val_libritts_clean, val_vctk)
        val_egs_cuts = [
            voicemos_main1_val.subset(first=10),
            val_vctk.subset(first=10),
            val_libritts_clean.subset(first=10),
        ]

        tr_libritts_clean = load_cuts(
            data_dir / "libritts" / "cuts_train-clean-360.jsonl.gz"
        )
        tr_vctk = load_cuts(data_dir / "vctk" / "cuts_train.jsonl.gz")
        # Split to equal size chunks if have favourite dataset use it multiple times for mux

        train_cuts = CutSet.mux(
            voicemos_main1_tr,
            tr_libritts_clean,
            tr_vctk,
            # num cuts in cutsets so their probability is proportional and are depleted around the same time
            weights=[len(voicemos_main1_tr), 116500, 38000],
        )
        test_cuts = voicemos_main1_test
    else:
        raise ValueError(f"Unknown data setup: {data_setup}")
    assert isinstance(val_egs_cuts, list), f"{type(val_egs_cuts)}"
    if train_ratio < 1.0:
        cutids = [c.id for c in train_cuts]
        num_cuts = math.ceil(train_ratio * len(cutids))
        cutids_chosen = np.random.choice(cutids, size=num_cuts, replace=False)
        logging.warning(
            f"Reduced training set from {len(cutids)=} to {len(cutids_chosen)=}"
        )
        train_cuts = train_cuts.subset(cut_ids=cutids_chosen)
    return train_cuts, val_cuts, val_egs_cuts, test_cuts


def get_noise_cuts(noise_data: str, min_noise_duration=5.0):
    # TODO add /lnet/express/data/hu-nonspeech/
    if noise_data == "musan_noise":
        # the noise label are cut ids
        # we are assuming long noise
        noise_cuts = CutSet.from_manifests(
            recordings=load_manifest(data_dir / "musan" / "recordings_noise.json")
        ).filter(lambda c: c.duration >= min_noise_duration)
    else:
        raise ValueError(f"Uknown noise data setup: {noise_data}")
    return noise_cuts


def load_annotator_stats(path: str):
    d = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            try:
                assert len(row) == 3, str(row)
                # See ./moosenet/bin/manifest_to_annotator_stats.py
                # Each row is annotator_id, annotator_correction, annotator_variance
                d[row[0]] = float(row[1]), float(row[2])

            except Exception as e:
                logging.error(f"Error on line {i} for file {path}")
                raise e
    return d
