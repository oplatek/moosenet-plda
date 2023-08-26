import lhotse
from lhotse import CutSet
from lhotse.cut import MixedCut, MonoCut
from lhotse.dataset import (
    DynamicBucketingSampler,
    BucketingSampler,
    SingleCutSampler,
    CutMix,
)
from lhotse.utils import ifnone
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, Tuple, List, Callable

from moosenet import data_dir
from moosenet.collate import (
    LoadAudio,
    PadAudio,
    CollateMOS,
    CollateEditDistance,
    CollatePitch,
    CollateSTOI_MCD,
    CollateMCD,
    CollateFbank,
    CollateWav2vecCTCtrn,
    CollatePhonesFromTrn,
    CollateNoiseLabels,
)


def get_train_dataloader(
    cuts: CutSet,
    dataset: Dataset,
    nworkers: int,
    prefetch_factor: float,
    nbuckets: int,
    max_batch_duration: float,
    min_cut_duration: float = 1.0,
    max_cut_duration: float = 12.0,
):
    # DynamicBucketing https://github.com/lhotse-speech/lhotse/pull/517
    # use DynamicBucketingSampler if problems with memory, it needs some time for estimating bucketing strategy
    sampler = DynamicBucketingSampler(
        cuts,
        max_duration=max_batch_duration,
        num_buckets=nbuckets,
        buffer_size=10000,
        shuffle=True,
        seed=42,
    )

    def filtercuts(c):
        return min_cut_duration < c.duration < max_cut_duration

    sampler.filter(filtercuts)
    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=None,
        num_workers=nworkers,
        worker_init_fn=lambda x: lhotse.set_caching_enabled(True),
        prefetch_factor=prefetch_factor,
        pin_memory=True,
    )


def get_val_dataloader(
    cuts: CutSet,
    dataset: Dataset,
    nworkers: int,
    prefetch_factor: float,
    max_batch_duration: float,
    min_cut_duration: float = 3.0,
    max_cut_duration: float = 12.0,
):
    sampler = SingleCutSampler(
        cuts, max_duration=max_batch_duration, shuffle=False, seed=42
    )

    def filtercuts(c):
        return min_cut_duration < c.duration < max_cut_duration

    sampler.filter(filtercuts)
    return DataLoader(
        dataset,
        batch_size=None,
        sampler=sampler,
        num_workers=nworkers,
        worker_init_fn=lambda x: lhotse.set_caching_enabled(True),
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        shuffle=False,
    )


def get_test_dataloader(
    cuts: CutSet,
    dataset: Dataset,
    nworkers: int,
    prefetch_factor: float,
    max_batch_duration: float,
):
    sampler = SingleCutSampler(
        cuts,
        max_duration=max_batch_duration,
        shuffle=False,
        seed=42,
        drop_last=False,
    )

    return DataLoader(
        dataset,
        batch_size=None,
        sampler=sampler,
        num_workers=nworkers,
        worker_init_fn=lambda x: lhotse.set_caching_enabled(True),
        pin_memory=True,
        prefetch_factor=prefetch_factor,
        shuffle=False,
    )


class MOS_VC_TTSDataset(Dataset):
    CLEAN_SOURCE = "clean_src_"

    def __init__(
        self,
        fbanks: CollateFbank,
        mos: CollateMOS,
        trn2phones: CollatePhonesFromTrn,
        pitch: CollatePitch,
        stoi_mcd: CollateSTOI_MCD,
        mcd: CollateMCD,
        noise_augment: CutMix,
        include_cuts=False,
        include_audios=False,
        wave_transforms: Optional[List[Callable[[CutSet], CutSet]]] = None,
        positive_cuts: float = 0.0,
    ):
        self.positive_cuts = positive_cuts
        self.noise_augment = noise_augment
        # on UFAL cluster causes memory corruption
        self.load_audio = LoadAudio(num_workers=0, fault_tolerant=True)
        self.wave_transforms = ifnone(wave_transforms, [])
        self.pad_audio = PadAudio()
        self.noise_label_snr = CollateNoiseLabels(self.noise_augment)
        self.stoi_mcd = stoi_mcd
        self.mcd = mcd
        self.fbanks = fbanks
        self.mos = mos
        self.trn2phones = trn2phones
        self.pitch = pitch
        # Cut transforms should not change MOS score interpretation

        self.include_cuts = include_cuts
        self.include_audios = include_audios

        self.sampling_rate = self.fbanks.sampling_rate
        assert (
            self.fbanks.sampling_rate == self.pitch.sampling_rate
        ), "Need to use different sampling rate in __getitem__"

    @property
    def num_listeners(self):
        """Number of unique listeners supported in training and inference"""
        return self.mos.num_annotators

    @property
    def noise_labels_idx2n(self):
        return self.noise_label_snr.idx2n

    @property
    def phones_trn_idx2p(self):
        return self.trn2phones.idx2p

    @property
    def fbanks_dim(self):
        return self.fbanks.feat_dim

    @property
    def pitch_dim(self):
        return self.pitch.feat_dim

    def empty_batch(self):
        batch = {
            "cuts": None,
            "audios": None,
            "num_cuts": None,
        }
        for k in (
            self.noise_label_snr.KEYS
            + self.stoi_mcd.KEYS
            + self.mcd.KEYS
            + self.pad_audio.KEYS
            + self.fbanks.KEYS
            + self.mos.KEYS
            + self.trn2phones.KEYS
            + self.pitch.KEYS
        ):
            batch[k] = None
        return batch

    @staticmethod
    def _prefix(d, prefix):
        return dict((f"{prefix}{k}", v) for k, v in d.items())

    def __getitem__(self, cuts: CutSet) -> dict:
        # We heavily rely on Lhotse caching mechanism when loading audio from disc
        # TODO check that it reads the audio truly only once

        # Sort the cuts by duration so that the first one determines the batch time dimensions.
        cuts = cuts.sort_by_duration(ascending=False)

        cuts = cuts.resample(self.sampling_rate)
        if self.positive_cuts <= 0.0:
            pos_cuts = CutSet.from_cuts([])
        elif self.positive_cuts < 1.0:
            pos_cuts = cuts.sample(max(int(len(cuts) * self.positive_cuts), 1))
        else:  # self.positive_cuts >= 1.0
            pos_cuts = cuts

        # Lhotse cuts.sample is INCONSISTENT :(
        # if n_positive == 1 it returns MonoCut instead of CutSet
        pos_cuts = (
            CutSet.from_cuts([pos_cuts])
            if not isinstance(pos_cuts, CutSet)
            else pos_cuts
        )

        for tfnm in self.wave_transforms:
            cuts = tfnm(cuts)
            pos_cuts = tfnm(pos_cuts)

        cuts = self.noise_augment(cuts)
        clean_src_cuts = [
            c.tracks[0].cut if isinstance(c, MixedCut) else c for c in cuts
        ]
        noise_cuts = [c for c in cuts if isinstance(c, MixedCut)]

        pos_audios, pos_cuts = self.load_audio(pos_cuts)

        # cuts are mix of MixedCuts and MonoCut
        noise_audios, noise_cuts = self.load_audio(noise_cuts)
        # The cut ids are the same
        clean_src_cuts = [
            c for c in clean_src_cuts if (c.id in noise_cuts or len(noise_cuts) == 0)
        ]
        clean_src_audios, clean_src_cuts = self.load_audio(clean_src_cuts)

        # Lets concatenate the cuts in batch
        # Clean utterances are first

        assert len(noise_cuts) == 0 or len(noise_cuts) == len(
            clean_src_cuts
        ), f"Use either noise_prob 0.0 or 1.0: {len(noise_cuts)} vs {len(clean_src_cuts)}"

        cuts = clean_src_cuts + noise_cuts + pos_cuts
        audios = clean_src_audios + noise_audios + pos_audios

        noise_d = self.noise_label_snr(cuts)
        fbank_d = self.fbanks(audios, cuts)

        fbanks, fbanks_lens = (
            fbank_d[CollateFbank.FBANK_FEATURES],
            fbank_d[CollateFbank.FBANK_FEATURES_LENS],
        )
        if fbanks is None:
            clean_fbanks, noisy_fbanks, clean_fbanks_lens, noisy_fbanks_lens = (
                None,
                None,
                None,
                None,
            )
        else:
            assert fbanks_lens is not None
            # WARNING: we treat positive_cuts as clean cuts
            clean_fbanks, noisy_fbanks = (
                fbanks[: -len(noise_cuts), ...],
                fbanks[-len(noise_cuts) :, ...],
            )
            clean_fbanks_lens, noisy_fbanks_lens = (
                fbanks_lens[: -len(noise_cuts), ...],
                fbanks_lens[-len(noise_cuts) :, ...],
            )

        real_batch = {
            "cuts": cuts if self.include_cuts else None,
            "audios": audios if self.include_audios else None,
            "num_clean_cuts": len(clean_src_cuts),
            "num_noisy_cuts": len(noise_cuts),
            "num_pos_cuts": len(pos_cuts),
            "num_cuts": len(cuts),
            # num_noise_cuts: len(noise_cuts) == num_cuts - num_clean_cuts
            **self.pad_audio(audios, cuts),
            **fbank_d,
            **noise_d,
            **self.stoi_mcd(clean_src_audios, noise_audios),
            **self.mcd(
                clean_fbanks, noisy_fbanks, clean_fbanks_lens, noisy_fbanks_lens
            ),
            **self.trn2phones(cuts),
            **self.mos(
                cuts,
                # used to determine which cuts which cuts are noisy
                snr=noise_d[CollateNoiseLabels.SNR],
            ),
            **self.pitch(cuts, audios),
        }
        batch = {**self.empty_batch(), **real_batch}
        return batch

    @classmethod
    def from_lhotse_cuts(
        cls,
        score_strategy: str,
        noise_cuts: CutSet,
        betas: Dict[str, float],
        dec_betas: Dict[str, float],
        model_name: str,
        snr_a=10,
        snr_b=20,
        noise_prob=1.0,
        # See losses and models and their defaults: DEFAULT_DEC_BETAS, DEFAULT_BETAS
        include_cuts=False,
        include_audios=False,
        wave_transforms: List[Callable[[CutSet], CutSet]] = None,
        positive_cuts: float = 0,
    ):

        collate_fbank = (
            model_name in ["ConformerFinalProjection", "ConformerFrameProjection"]
            or dec_betas["mcd"] > 0.0
        )
        collate_phones = betas["ctc_phntrn"] > 0.0
        collate_pitch = betas["pitch"] > 0.0
        collate_stoi = betas["stoi"] > 0.0 or dec_betas["stoi"]
        collate_mcd = dec_betas["mcd"] > 0.0

        assert not collate_mcd or collate_fbank, "MCD uses fbanks for calculation"
        fbank = CollateFbank(dummy=not collate_fbank)
        if score_strategy in [
            CollateMOS.STRATEGY_RANDOM,
            CollateMOS.STRATEGY_RANDOM_AVG,
        ]:
            annotator2int = CollateMOS.load_annotator2int()
        else:
            annotator2int = None

        return cls(
            fbank,
            CollateMOS(
                strategy=score_strategy,
                annotator2int=annotator2int,
            ),
            CollatePhonesFromTrn(dummy=not collate_phones),
            CollatePitch(sampling_rate=16000, dummy=not collate_pitch),
            CollateSTOI_MCD(
                sampling_rate=16000,
                dummy_stoi=not collate_stoi,
                # Espnet implementation of MCD which calculates MFCC from audio is slow
                dummy_mcd=True,
            ),
            # 41 Mel Fbanks - inspiration from Espnet
            CollateMCD(dummy=not collate_mcd, n_fbank=fbank.num_mel_bins, n_mfcc=41),
            noise_augment=CutMix(
                noise_cuts,
                snr=[snr_a, snr_b],
                prob=noise_prob,
                preserve_id=True,
                pad_to_longest=False,
            ),
            wave_transforms=wave_transforms,
            include_cuts=include_cuts,
            include_audios=include_audios,
            positive_cuts=positive_cuts,
        )
