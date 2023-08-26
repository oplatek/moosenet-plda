import torch
import csv
import numpy as np
from torch.nn import CrossEntropyLoss
from copy import deepcopy
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility
from pathlib import Path
from lhotse.utils import Pathlike, ifnone, LOG_EPSILON, compute_num_frames, fastcopy
import torch.nn.functional as F
import torchaudio.functional as TAF
import torchaudio.transforms as T
from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Optional, List, Callable, Type, Dict, Tuple
from statistics import mean, median, variance
from lhotse import CutSet, Recording, Fbank, FbankConfig
from lhotse.cut import MixedCut
import random
import logging
from lhotse.features import FeatureExtractor
from lhotse.dataset import OnTheFlyFeatures
from lhotse.dataset.input_strategies import _get_executor, ExecutorType
from math import ceil
from lhotse.dataset.collation import (
    collate_vectors,
    collate_matrices,
    read_audio_from_cuts,
)
import torchaudio.pipelines as pipelines

from moosenet.mcd2 import MelCepstralDistortion
from moosenet import data_dir
from moosenet.modules import GreedyCTCDecoder
from moosenet.edit_dist import batch_edit_distance
import g2p_en


class CollateIsHuman:
    ISHUMAN = "source_kind"
    LABEL_HUMAN = "MOOSE_HUMAN"
    LABEL_SYSTEM = "MOOSE_SYS"
    LABEL_SYSHUM = "MOOSE_SYSHUM"  # unknown label

    KEYS = (ISHUMAN,)

    def __init__(self, skip_humsys_prob=0.1):
        self.skip_humsys_prob = skip_humsys_prob
        self.idx2ish = [self.LABEL_HUMAN, self.LABEL_SYSTEM, self.LABEL_SYSHUM]
        self.ish2idx = dict((ish, i) for i, ish in enumerate(self.idx2ish))

    def __call__(self, cuts):
        ishumans = []
        raise NotImplementedError(
            "implement heuristics for dataset to tell if it is human or not"
        )
        B = len(cuts)
        for c in cuts:
            # TODO determine if human or system and apply propbability for masking
            # TODO use self.skip_humsys_prob
            source_label = self.LABEL_SYSHUM
            ishumans.append(self.ish2idx[source_label])
        return {
            self.ISHUMAN: torch.LongTensor(ishumans).reshape(B, 1),
            self.ISHUMAN_LENS: torch.BoolTensor([True] * B).reshape(B),
        }


class CollateNoiseLabels:
    LABEL = "noise_labels"
    SNR = "noise_snr"
    NOISE_LENS = "noise_lens"
    # TODO should I add dataset labels?
    # dataset labels

    KEYS = (LABEL, SNR, NOISE_LENS)

    # label of "noise" saying the audio is kept as an original
    MOOSE_NO_AUGMENT = "MOOSE_NO_AUGMENT"
    # SNR is decibels 0.0 equal signal and noise
    # SNR 100 means that clean audio shoudl be 100dB lauder than noise (which is missing)
    SNR_NO_AUGMENT = 100.0

    def __init__(self, mixed_cuts):
        # TODO extract proper labels not cut ids!
        self.idx2n = [self.MOOSE_NO_AUGMENT] + [c.id for c in mixed_cuts.cuts]
        self.n2idx = dict((cid, i) for i, cid in enumerate(self.idx2n))

    def __call__(self, cuts):
        B = len(cuts)

        snrs = []
        labels = []
        for c in cuts:
            if not isinstance(c, MixedCut):
                label = self.MOOSE_NO_AUGMENT
                snr = self.SNR_NO_AUGMENT
            else:
                # expecting MixedCut - noise augmentation was run
                # TODO extract proper labels not cut ids!
                label = c.tracks[1].cut.id
                snr = c.tracks[1].snr

            labels.append(self.n2idx[label])
            snrs.append(snr)

        return {
            self.LABEL: torch.LongTensor(labels).reshape(B, 1),
            self.SNR: torch.FloatTensor(snrs).reshape(B, 1),
            # Always using the labels ATM
            self.NOISE_LENS: torch.BoolTensor([True] * B).reshape(B),
        }


class CollateMCD:
    MCD = "MCD2"
    KEYS = (MCD,)

    def __init__(self, dummy=False, n_fbank=None, n_mfcc=None):
        self.dummy = dummy

        assert dummy or (n_fbank is not None and n_mfcc is not None), str(
            (n_fbank, n_mfcc)
        )

        if not self.dummy:
            self.mcd = MelCepstralDistortion(n_fbank=n_fbank, n_mfcc=n_mfcc)

    def __call__(self, logfbanks, aug_logfbanks, logf_lens, aug_logf_lens):
        if self.dummy:
            return {self.MCD: None}

        assert logfbanks.shape[0] == aug_logfbanks.shape[0], str(
            (logfbanks.shape, aug_logfbanks.shape)
        )
        B_half = logfbanks.shape[0]

        mcd = self.mcd.update(aug_logfbanks, logfbanks, aug_logf_lens, logf_lens)
        mcd = torch.FloatTensor(mcd).reshape(B_half, 1)
        # TODO "normalize" MCD metric to MOS range assuming that orig has score 5 for HUMAN
        # or cut.MOS or predicted MOS from NN
        # Best value is 0 (assuming that original features - the audio is perfect)
        identical_value = torch.full((B_half, 1), 0.0)
        mcd = torch.cat((identical_value, mcd), 0)

        return {self.MCD: mcd}


class CollatePESQ:
    """
    PESQ returns score in interval [-0.5, 4.5] the higher the better.
    See https://github.com/ludlows/PESQ
    """

    pass


class CollateSTOI_MCD:
    """For STOI see https://ceestaal.nl/code/
    and https://torchmetrics.readthedocs.io/en/stable/audio/short_time_objective_intelligibility.html

    STOI is an Intelligibility measure which is highly correlated with the intelligibility
    of degraded speech signals, e.g., due to additive noise, single/multi-channel noise reduction,
    binary masking and vocoded speech as in CI simulations.
    The STOI-measure is intrusive, i.e., a function of the clean and degraded speech signals.
    STOI may be a good alternative to the speech intelligibility index (SII) or the speech transmission index (STI),
    when you are interested in the effect of nonlinear processing to noisy speech, e.g.,
    noise reduction, binary masking algorithms, on speech intelligibility. (Cees Taal)
    """

    STOI = "STOI"
    MCD = "MCD"
    KEYS = (STOI, MCD)

    def __init__(
        self,
        sampling_rate,
        dummy_stoi=False,
        dummy_mcd=False,
        n_fbank=None,
        n_mfcc=None,
    ):
        self.dummy_stoi = dummy_stoi
        self.dummy_mcd = dummy_mcd
        self.sampling_rate = sampling_rate
        # TODO first add also PESQ

    def __call__(self, orig_audios, aug_audios):
        stoi, mcd = None, None
        if self.dummy_stoi and self.dummy_mcd:
            return {self.STOI: stoi, self.MCD: mcd}

        assert len(orig_audios) == len(
            aug_audios
        ), f"{orig_audios.shape} vs {aug_audios.shape}"
        B_half = len(orig_audios)
        # B = B_half * 2 ; Batch size is double size of the vector we return

        # trim the mixed audio to the original size
        aug_audios = [aa[: oa.shape[0]] for aa, oa in zip(aug_audios, orig_audios)]

        if not self.dummy_stoi:
            stoi = [
                short_time_objective_intelligibility(aa, oa, self.sampling_rate)
                for aa, oa in zip(aug_audios, orig_audios)
            ]
            stoi = torch.FloatTensor(stoi).reshape(B_half, 1)
            # TODO "normalize" STOI metric to MOS range assuming that orig has score 5 for HUMAN
            # or cut.MOS or predicted one from NN
            stoi = torch.cat((torch.ones(B_half, 1), stoi), 0)

        if not self.dummy_mcd:
            from moosenet.mcd import calculate_mcd

            # WARNING: this mcd implementation is slow because it computes MFCC features from audio again
            # Plus frame_shift ie hopsize is 5ms
            # See CollateMCD for alternative implementation
            # Keeping it here are reference
            #
            # TODO "normalize" MCD metric to MOS range assuming that orig has score 5 for HUMAN
            # or cut.MOS or predicted MOS from NN
            mcd = [
                calculate_mcd(oa, aa, self.sampling_rate)
                for aa, oa in zip(aug_audios, orig_audios)
            ]
            mcd = torch.FloatTensor(mcd).reshape(B_half, 1)
            mcd = torch.cat((torch.zeros(B_half, 1), mcd), 0)

        return {self.STOI: stoi, self.MCD: mcd}


# TODO Use World Pitch as features
class CollatePitch:
    PITCHES = "pitches"
    PITCHES_LENS = "pitches_lens"
    PAD = 0.0
    KEYS = (PITCHES, PITCHES_LENS)

    def __init__(self, sampling_rate, frame_shift=10, dummy=False):
        self.dummy = dummy
        self.sampling_rate = sampling_rate
        # FRAME_SHIFT 40 ms is hack it is 10 ms Fbank shift times SUBSAMPLING_FACTOR of Conformer
        self.frame_shift = 40.0  # ms

    @property
    def feat_dim(self):
        return 2

    def __call__(self, cuts, audios):
        if self.dummy:
            return {self.PITCHES: None, self.PITCHES_LENS: None}
        pitch_tensors = []
        pitch_lens = []
        for audio in audios:
            # TODO add alternative pitch also?
            # pitch = TAF.detect_pitch_frequency(audio, self.sampling_rate)

            # returns (B, Frames, 2) if batched
            # but here it returns (Frames, 2)
            kaldi_pitch = TAF.compute_kaldi_pitch(
                audio,
                self.sampling_rate,
                frame_shift=self.frame_shift,
                snip_edges=False,
            )
            pitch_tensors.append(kaldi_pitch)
            pitch_lens.append(kaldi_pitch.shape[0])

        return {
            self.PITCHES: collate_matrices(pitch_tensors, padding_value=self.PAD),
            self.PITCHES_LENS: torch.LongTensor(pitch_lens),
        }


class CollatePhonesFromTrn:
    PHNS_GOLDTRN = "phns_goldtrn"
    PHNS_GOLDTRN_LENS = "phns_goldtrn_lens"
    KEYS = (PHNS_GOLDTRN, PHNS_GOLDTRN_LENS)

    def __init__(self, dummy=False):
        self.dummy = dummy
        # just extact the mappings and let espnet deal with multiprocessing issues
        g2p_tmp = g2p_en.G2p()
        self.p2idx = g2p_tmp.p2idx
        self.idx2p = g2p_tmp.idx2p
        del g2p_tmp
        self._g2p_en = None  # g2p is not picklable - do not instantiate before forking - do not use self.g2p

        self.PADIDX = self.p2idx["<pad>"]
        self.UNKIDX = self.p2idx["<unk>"]
        self.BOS = self.p2idx["<s>"]
        self.EOS = self.p2idx["</s>"]
        self.SPACE = "|"
        self.p2idx[self.SPACE] = len(self.p2idx)
        self.idx2p[len(self.idx2p)] = self.SPACE
        assert len(self.p2idx) == len(self.idx2p)

    def _g2p(self, text):
        if self._g2p_en is None:
            self._g2p_en = g2p_en.G2p()

        phones = self._g2p_en(text.strip())
        # treat special tokens/interpuction and ' ' as SPACE aka |
        phones = [p if p in self.p2idx else self.SPACE for p in phones]
        # remove duplicate well ... not good -> well|||||not|good -> wel|not|good
        phones = (
            self.SPACE.join(
                [
                    run
                    for run in "/".join(phones).strip(self.SPACE).split(self.SPACE)
                    if run != "/"
                ]
            )
            .strip("/")
            .split("/")
        )

        return phones

    def __call__(self, cuts):
        if self.dummy:
            return {self.PHNS_GOLDTRN: None, self.PHNS_GOLDTRN_LENS: None}
        token_ids = []
        token_ids_lens = []
        for c in cuts:
            text = c.supervisions[0].text if c.supervisions else None
            if text is None or len(text) == 0:
                token_ids.append([])
                token_ids_lens.append(0)
            else:
                tokens = self._g2p(text)
                token_ids.append([self.p2idx[t] for t in tokens])
                token_ids_lens.append(len(tokens))

        MT = max(token_ids_lens)
        pad_token_ids = [ids + ([self.PADIDX] * (MT - len(ids))) for ids in token_ids]
        return {
            self.PHNS_GOLDTRN: torch.LongTensor(pad_token_ids),
            self.PHNS_GOLDTRN_LENS: torch.LongTensor(token_ids_lens),
        }


class LoadAudio:
    def __init__(
        self,
        num_workers: int = 0,
        fault_tolerant: bool = False,
        executor_type: Type[ExecutorType] = ThreadPoolExecutor,
    ) -> None:
        """Inspired by lhotse.dataset.input_strategy.AudioSamples
        Read audios for all cuts in batch so the calller
        can postprocess the audio.
        Convenient if multiple collation uses the same audio samples
        """
        self.num_workers = num_workers
        self._executor_type = executor_type
        self.fault_tolerant = fault_tolerant

    def __call__(self, cuts):
        audios, read_cuts = read_audio_from_cuts(
            cuts,
            executor=_get_executor(self.num_workers, executor_type=self._executor_type),
            suppress_errors=self.fault_tolerant,
        )

        return audios, read_cuts


class PadAudio:
    PAD_AUDIO = "pad_audio"
    PAD_AUDIO_LENS = "pad_audio_lens"
    KEYS = (PAD_AUDIO, PAD_AUDIO_LENS)

    @staticmethod
    def repeat_wav(wavs: List[torch.Tensor]):
        max_len = max(wavs, key=lambda x: x.shape[0]).shape[0]
        output_wavs = []
        for wav in wavs:
            amount_to_pad = max_len - wav.shape[0]
            padding_tensor = wav.repeat(1 + amount_to_pad // wav.shape[0])
            padded = torch.cat((wav, padding_tensor[:amount_to_pad]))
            output_wavs.append(padded)
        return torch.stack(output_wavs, dim=0)

    def __call__(self, audios, cuts):
        audio_lens = torch.tensor([cut.num_samples for cut in cuts], dtype=torch.int32)
        # TODO try different padding value for mean and max pooling
        # default padding value
        # audios = collate_vectors(audios, padding_value=CrossEntropyLoss().ignore_index)
        # Let's use 0 for silence in wavs
        # audios = collate_vectors(audios, padding_value=0.0)
        audios = self.repeat_wav(audios)
        return {self.PAD_AUDIO: audios, self.PAD_AUDIO_LENS: audio_lens}


class CollateFbank:
    FBANK_FEATURES = "fbank_features"
    FBANK_FEATURES_LENS = "fbank_features_lens"
    KEYS = (FBANK_FEATURES, FBANK_FEATURES_LENS)

    def __init__(
        self, feat_config: Optional[Pathlike] = None, dummy=False, num_mel_bins=80
    ):
        self.dummy = dummy

        self.extractor = Fbank(FbankConfig(num_filters=num_mel_bins))
        try:
            logging.info(f"Sample rate {self.sampling_rate}, feat_dim {self.feat_dim}")
        except Exception as e:
            logging.exception("Failed to compute properties sampling_rate, feat_dim")
            raise e

        if dummy:
            return

        self.input_strategy = OnTheFlyFeatures(self.extractor)

    @property
    def sampling_rate(self):
        return self.extractor.config.sampling_rate

    @property
    def frame_shift(self):
        return self.extractor.config.frame_shift

    @property
    def hop_size(self):
        return self.frame_shift

    @property
    def num_mel_bins(self):
        return self.extractor.config.num_filters

    @property
    def feat_dim(self):
        return self.extractor.feature_dim(self.sampling_rate)

    def __call__(self, audios, cuts):
        """See https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/input_strategies.py#L333"""  # noqa
        if self.dummy:
            # Let the model handle the Nones
            return {self.FBANK_FEATURES: None, self.FBANK_FEATURES_LENS: None}

        assert all(c.sampling_rate == cuts[0].sampling_rate for c in cuts)
        features_single = self.extractor.extract_batch(
            audios, sampling_rate=cuts[0].sampling_rate
        )
        features = collate_matrices(features_single, padding_value=LOG_EPSILON)
        features_lens = torch.tensor(
            [
                compute_num_frames(
                    cut.duration, self.extractor.frame_shift, cut.sampling_rate
                )
                for cut in cuts
            ],
            dtype=torch.int32,
        )

        return {self.FBANK_FEATURES: features, self.FBANK_FEATURES_LENS: features_lens}


class CollateWav2vecCTCtrn:
    """Implemented according to
    https://pytorch.org/audio/main/tutorials/asr_inference_with_ctc_decoder_tutorial.html#preparation  # noqa
    params:
        name  See constants in the _wav2vec2/impl.py
    """

    W2V_FEATS = "w2v_feats"
    W2V_FEATS_LENS = "w2v_feats_lens"
    W2V_CHAR_IDXS = "w2v_char_idxs"
    W2V_CHAR_IDXS_LENS = "w2v_char_idxs_lens"

    @classmethod
    def dummyCollate(cls):
        class DummyCollate:
            KEYS = ()

            def __call__(*args):
                logging.debug(f"DummyCollateWav2vecCTCtrn args: {args}")
                return {}

            @property
            def labels(self):
                return ["DUMMY"]

            @property
            def sampling_rate(self):
                return 16000  # hardcoded hack

        return DummyCollate()

    @property
    def KEYS(self):
        return (
            f"{self.prefix}{self.W2V_FEATS}",
            f"{self.prefix}{self.W2V_FEATS_LENS}",
            f"{self.prefix}{self.W2V_CHAR_IDXS}",
            f"{self.prefix}{self.W2V_CHAR_IDXS_LENS}",
        )

    def __init__(self, bundle_name, prefix: Optional[str] = None):
        self.bundle_name = bundle_name
        self.bundle = getattr(pipelines, bundle_name)
        self.prefix = ifnone(prefix, "")

        self.acoustic_model = self.bundle.get_model()

        # Blank is hardcoded as 0. See
        # https://github.com/pytorch/audio/blob/a1dc9e0ae65edde83361bfdc255f2db416ede139/torchaudio/pipelines/_wav2vec2/impl.py#L150  # noqa
        self.greedy_dec = GreedyCTCDecoder(self.labels, blank=0)
        self.int2token = dict([(i, t) for i, t in enumerate(self.labels)])

    def tokens2strings(self, indices, lens, token_sep=None):
        token_sep = ifnone(token_sep, "")
        hyps = []
        for b in range(len(lens)):
            d = lens[b]
            hyp = []
            for i in range(d):
                hyp.append(self.int2token[indices[b, i].item()])
            hyps.append(token_sep.join(hyp))
        return hyps

    @property
    def sampling_rate(self):
        return self.bundle.sample_rate

    @property
    def labels(self):
        return self.bundle.get_labels()

    def __call__(self, cuts, audios):
        # See https://github.com/pytorch/audio/blob/a1dc9e0ae65edde83361bfdc255f2db416ede139/torchaudio/models/wav2vec2/model.py#L84  # noqa

        waveforms = collate_vectors(audios, padding_value=0.0)  # pad with silence
        lens = torch.LongTensor([audio.shape[0] for audio in audios])
        # TODO use it
        # # feats, feats_lens = self.acoustic_model.extract_features(
        # #     waveforms, lengths=lens
        # # )
        feats, feats_lens = None, None

        emission, _ = self.acoustic_model(waveforms, lengths=lens)
        idxs, idx_lens = self.greedy_dec(emission)

        return {
            f"{self.prefix}{self.W2V_FEATS}": feats,
            f"{self.prefix}{self.W2V_FEATS_LENS}": feats_lens,
            f"{self.prefix}{self.W2V_CHAR_IDXS}": idxs,
            f"{self.prefix}{self.W2V_CHAR_IDXS_LENS}": idx_lens,
        }


class CollateEditDistance:
    EDIT_DISTANCE: torch.LongTensor = "edit_distance"  # (B, 1)
    EDIT_DISTANCE_LENS: torch.BoolTensor = "edit_distnace_len"  # B

    @classmethod
    def KEYS(cls, prefix):
        return f"{prefix}{cls.EDIT_DISTANCE}"

    def __init__(self, prefix):
        self.prefix = prefix

    def __call__(self, A, A_lens, B, B_lens):
        """
        :param A: torch.tensor of shape (batch, max_seq_len) of decoded hypotheses
        :param B: torch.tensor of shape (batch, max_seq_len) of reference hypotheses
        """

        # ref_idxs, ref_idx_lens = self.greedy_dec(ref_emission)

        edit_distances = batch_edit_distance(A, A_lens, B, B_lens)
        edit_distances_lens = B_lens > 0
        return {
            f"{self.prefix}{self.EDIT_DISTANCE}": edit_distances,
            f"{self.prefix}{self.EDIT_DISTANCE_LENS}": edit_distances_lens,
        }


class CollateMOS:
    # make it stupidly large to notice if we masked incorrectly
    PAD = -9999999999

    MOS_FINAL = "mos_final"
    SCORE_FINAL = "A_score"
    MOS_VAR_FINAL = "mos_var"
    ANNOTATOR_ID = "A_id"
    MOS_FINAL_MASK = "mos_final_mask"
    MOS_DEDUCTED_MASK = "mos_deducted_mask"
    KEYS = (
        MOS_FINAL,
        MOS_VAR_FINAL,
        SCORE_FINAL,
        ANNOTATOR_ID,
        MOS_FINAL_MASK,
        MOS_DEDUCTED_MASK,
    )
    STRATEGY_MEAN = "mean"
    STRATEGY_RANDOM = "random"
    STRATEGY_RANDOM_AVG = "random_avg"
    STRATEGY_MODES = [STRATEGY_MEAN, STRATEGY_RANDOM, STRATEGY_RANDOM_AVG]
    ANNOTATOR_MEAN = "moose_avg"

    LISTENER_ID_SEP = "_listenerID_"
    LISTENER_ID_FILEPATH = "data/bvcc/listener_map.csv"

    def __init__(
        self,
        strategy=None,
        seed: Optional[int] = None,
        subsampling_factor: Optional[float] = None,
        annotator2int: Optional[Dict[str, int]] = None,
    ):
        self.seed = ifnone(seed, 42)
        self.annotator_rng = random.Random(self.seed)
        self.strategy = ifnone(strategy, "mean")
        assert self.strategy in self.STRATEGY_MODES, str(self.strategy)

        # WARNING - in your models expect that if len(annotator2int) == 1
        # (or zero) you should not need to use annotator embedding
        annotator2int = ifnone(annotator2int, {self.ANNOTATOR_MEAN: 0})
        assert (
            annotator2int[self.ANNOTATOR_MEAN] == 0
        ), "The average annotator should always have id 0"
        if len(annotator2int) <= 1:
            logging.warning(
                f"Using ONLY {len(annotator2int)} so you probably should not model listeners"
            )

        self.annotator2int = annotator2int

        from moosenet.models import SUBSAMPLING_FACTOR

        self.subsampling_factor = ifnone(subsampling_factor, SUBSAMPLING_FACTOR)

    @property
    def num_annotators(self) -> int:
        return len(self.annotator2int)

    @classmethod
    def load_annotator2int(cls, listener_file: Optional[str] = None):
        listener_file = Path(ifnone(listener_file, cls.LISTENER_ID_FILEPATH))
        assert (
            listener_file.exists()
        ), f"Prepare {listener_file} by running 'moosenet listener-map data/bvcc/cuts_* {listener_file}'"
        with open(listener_file, "r") as csvfile:
            annotator2int = dict((a, int(i)) for a, i in csv.reader(csvfile))
        return annotator2int

    @staticmethod
    def cuts2annotators(cuts: CutSet):
        annotators = set()
        for c in cuts:
            for k in c.supervisions[0].custom.get("MOS", {}).keys():
                annotators.add(k)
        return annotators

    @staticmethod
    def system_name(wav_name):
        return wav_name.split("-")[0]

    @classmethod
    def system_names(cls, cuts):
        return [cls.system_name(c.id) for c in cuts]

    @staticmethod
    def MOS(batch):
        """
        Returns MOS scores for each cut in a batch.

        MOS is dictionary of {'Listener_ID1': integer_1_5_score, 'Listerner_ID2': integer_1_5_score, etc}
        """
        return [
            c.supervisions[0].custom.get("MOS", {}) if c.supervisions else {}
            for c in batch["cuts"]
        ]

    @classmethod
    def cutScores2cutsScore(cls, c):
        """Converts single cut with audio and set of opinion scores
        into set of cuts with identical audio and single opinion score
        """
        mosd = c.supervisions[0].custom.get("MOS", {})
        if len(mosd) <= 1:
            yield c
            return

        assert len(mosd) > 1, str(mosd)
        for lid, s in mosd.items():
            c_one = fastcopy(c, id=f"{c.id}{cls.LISTENER_ID_SEP}{lid}")
            c_one.supervisions[0].custom["MOS"] = {lid: s}
            yield c_one

    def _collate_mos(self, mosd):

        scores = mosd.values()
        mos = mean(scores)
        mos_variance = variance(scores, mos)

        if self.strategy == "mean":
            annotator_id, score = CollateMOS.ANNOTATOR_MEAN, mos
        if self.strategy == "random":
            annotator_id, score = self.annotator_rng.choice(list(mosd.items()))
        elif self.strategy == "random_avg":
            annotator_id, score = self.annotator_rng.choice(list(mosd.items()))
            # choose from one of the annotators or use AVG annotator embedding for training
            if self.annotator_rng.random() < 1 / (len(mosd) + 1):
                annotator_id, score = self.ANNOTATOR_MEAN, mos

        return (
            mos,
            mos_variance,
            score,
            self.annotator2int[annotator_id],
        )

    def __call__(
        self,
        cuts: CutSet,
        snr: Optional[torch.Tensor] = None,
    ):
        B = len(cuts)
        (
            mos_final,
            mos_variances,
            ann_scores,
            annotator_ids,
            mos_final_mask,
            mos_deducted_mask,
        ) = ([], [], [], [], [], [])
        for i, cut in enumerate(cuts):
            mosd = (
                cut.supervisions[0].custom.get("MOS", None)
                if cut.supervisions
                else None
            )

            corrupted_with_noise = (
                snr is not None and snr[i] != CollateNoiseLabels.SNR_NO_AUGMENT
            )

            # TODO add if is not currupted_with_noise and IS_HUMAN then 5.0
            if mosd is None or len(mosd) == 0:
                logging.info(f"cut {cut.id} has no MOS score!")
                # one should ignore this cut for training with MOS score
                asc, aid, mv, m, mm, md = 0.0, 0, 0.0, self.PAD, 0, 0
            elif corrupted_with_noise:
                #
                # WARNING!!!!!
                #
                # We assume that every noise damages the audio so the MOS score would be 1.0
                # however in the mask we say that this is not valid
                # we use mask_deducted to say we are fairly certain that this MOS + score should be 1
                asc, m, mm, md = 1, 1.0, 0, 1
                aid, mv = 0, 0.0
            else:
                mm, md = 1, 0  # mos score is GT not deducted from augmentation
                m, mv, asc, aid = self._collate_mos(mosd)

            mos_final.append(m)
            mos_variances.append(mv)
            ann_scores.append(asc)
            annotator_ids.append(aid)
            mos_final_mask.append(mm)
            mos_deducted_mask.append(md)

        annotator_ids = torch.IntTensor(annotator_ids).reshape(B, 1)
        mos_variances = torch.FloatTensor(mos_variances).reshape(B, 1)
        mos_final = torch.FloatTensor(mos_final).reshape(B, 1)
        ann_scores = torch.FloatTensor(ann_scores).reshape(B, 1)
        mos_final_mask = torch.BoolTensor(mos_final_mask)
        mos_deducted_mask = torch.BoolTensor(mos_deducted_mask)

        return {
            self.MOS_FINAL: mos_final,
            self.MOS_VAR_FINAL: mos_variances,
            self.SCORE_FINAL: ann_scores,
            self.ANNOTATOR_ID: annotator_ids,
            self.MOS_FINAL_MASK: mos_final_mask,
            self.MOS_DEDUCTED_MASK: mos_deducted_mask,
        }
