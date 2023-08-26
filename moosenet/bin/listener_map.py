import click
import csv
from moosenet.bin import moosenet
from lhotse import CutSet
from typing import List
from moosenet.datasetup import load_cuts
from moosenet.collate import CollateMOS


@moosenet.command()
@click.argument("cutsets", nargs=-1)  # Use all cutsets you will use for speakers
@click.argument("out")
@click.option(
    "--num_listeners",
    type=int,
    default=3000,
)
# help="Maximum number of listeners to support - has to greater than number of listeners in the cutsets",
def listener_map(cutsets: List[CutSet], out: str, num_listeners: int):
    cuts = load_cuts(*cutsets)
    ann_set = CollateMOS.cuts2annotators(cuts)
    n = len(ann_set) + 1
    assert n <= num_listeners, f"{n} vs {num_listeners}"
    assert CollateMOS.ANNOTATOR_MEAN not in ann_set
    # ANNOTATOR_MEAN is mapped to idx 0
    annotators = [CollateMOS.ANNOTATOR_MEAN]
    annotators.extend(a for a in ann_set)
    for k in range(n, num_listeners):
        annotators.append(f"RESERVED_{k}")
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows((a, i) for i, a in enumerate(annotators))
