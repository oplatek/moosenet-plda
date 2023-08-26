import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from typing import Dict, List
from lhotse.dataset.collation import collate_vectors
from torch.optim.optimizer import Optimizer
from moosenet.utils import length_to_mask
import torch.optim.lr_scheduler


class LabelsProjection(nn.Module):
    def __init__(self, input_dim, n_labels, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.n_labels = n_labels
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(input_dim, n_labels)

    def forward(self, x):
        scores = self.proj(self.dropout(x))
        # See https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html#torch.nn.CTCLoss
        probs = F.log_softmax(scores, dim=-1)
        return probs


class ScoreProjection(nn.Module):
    def __init__(
        self,
        input_dim,
        out_dim,
        dropout=0.3,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.lin_projection = nn.Linear(input_dim, out_dim)

    def forward(self, x):
        return self.lin_projection(self.dropout(x))


class MOSScoreProjection(nn.Module):
    def __init__(
        self,
        input_dim,
        range_epsilon=0.01,
        min_value=1.0,
        max_value=5.0,
        dropout=0.3,
    ):
        """
        Projecting input vector to scalar with range [1-eps, 5 + eps]
        """
        super().__init__()
        self.ffn = torch.nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, 1),
            nn.Tanh(),
        )
        self.TWO_with_eps = 2.0 + range_epsilon
        self.middle = (min_value + max_value) / 2  # 3 for MOS

    def forward(self, x):
        # Epsilon makes sure we can output the min and max MOS values
        # function range is [1-eps, 5 + eps] for MOS
        return (self.ffn(x) * self.TWO_with_eps) + self.middle


class BiLSTMDecoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.lstm_hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

    @property
    def output_dim(self):
        # We use bidirectional LSTM -> times two
        return 2 * self.lstm_hidden_dim

    @property
    def hidden_dim(self):
        # Lstm returns hidden state of shape (2 * num_layers, hidden_dim, feature_dim)
        # 2 is for bidirectional
        return 2 * self.num_layers * self.lstm_hidden_dim

    def forward(self, x, lens):
        """
        x: shape (B, T, F)
        """
        # The "frame" scores are not frame scores there are encoder step scores
        BT_lstm_outputs, (hidden, _cell) = self._iter_forward(x, lens)
        # (B, T, lstm_hidden_dim * 2) -> (B, lstm_hidden_dim * 2)
        lstm_final_out = self._final_out(BT_lstm_outputs, lens)
        return lstm_final_out

    def _iter_forward(self, x, lens):
        # Docs:
        #  https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        # Inspiration from
        #  https://stackoverflow.com/questions/51030782/why-do-we-pack-the-sequences-in-pytorch
        x = torch.nn.utils.rnn.pack_padded_sequence(
            x, lens.cpu(), batch_first=True, enforce_sorted=False
        )
        x, hidden_cell_states = self.lstm(x)
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return x, hidden_cell_states

    def _final_out(self, output, lens):
        assert output.shape[2] == 2 * self.lstm_hidden_dim, str(
            (output.shape, self.lstm_hidden_dim)
        )
        last_idx = (lens.long() - 1).squeeze()
        # forward takes the first half of the last dim
        out_forward = output[range(output.shape[0]), last_idx, : self.lstm_hidden_dim]

        # reverse takes the first half of the last dim
        # in reverse the first time ouput is the final
        out_reverse = output[:, 0, self.lstm_hidden_dim :]
        assert out_forward.shape == out_reverse.shape, str(
            (out_forward.shape, out_reverse.shape)
        )
        # TODO fix for non frame version with https://github.com/sarulab-speech/UTMOS22/blob/756fdf28785386fbfdc082c3fef0f35ab1b59c3c/strong/model.py#L90  # noqa
        return torch.cat((out_forward, out_reverse), 1)


class MaxAvgPoolNN(torch.nn.Module):
    def __init__(self, input_dim, out_dim, num_layers=2, dropout=0.3, init_lin=None):
        super().__init__()
        self.out_dim = out_dim

        assert input_dim == out_dim or num_layers > 0, (
            "Use at least one layer for projection different dimension sizes!"
            f"{input_dim} vs {out_dim} for {num_layers} layers"
        )

        lin_layer = [nn.Dropout(dropout), nn.Linear(out_dim, out_dim), nn.ReLU()]
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Sequential(nn.Linear(input_dim, out_dim), nn.ReLU())
            if num_layers > 0
            else nn.Identity(),
            *(lin_layer * (num_layers - 1)),
        )

        if init_lin == "linspace":
            self.apply(self._lin_init_linspace)
        elif init_lin == "one":
            self.apply(self._lin_init_one)
        else:
            # for None keep original init params
            assert init_lin is None, f"Unknown special init {init_lin}"

    def forward(self, x, lens):
        """
        x: shape (B, T, F)
        gloval average and max pooling with masking + MLP
        """
        # masked input
        m = length_to_mask(lens)
        x[~m] = 0.0
        # mean
        maxval, _indices = torch.max(x, dim=1)
        x = torch.sum(x, dim=1) / lens.unsqueeze(dim=1)
        # mean + max global pooling
        x = x + maxval
        # mlp
        x = self.mlp(x)

        return x

    @staticmethod
    def _lin_init_linspace(m):
        if isinstance(m, nn.Linear):
            # weight
            n_weight_params = np.prod(
                [m.weight.size(dim=d) for d in range(m.weight.dim())]
            )
            weight_a = torch.arange(start=1, end=2, step=1 / n_weight_params)
            m.weight = torch.nn.Parameter(weight_a.reshape(m.weight.shape))
            # bias
            n_bias_params = np.prod([m.bias.size(dim=d) for d in range(m.bias.dim())])
            bias_a = torch.arange(start=3, end=4, step=1 / n_bias_params)
            m.bias = torch.nn.Parameter(bias_a.reshape(m.bias.shape))

    @staticmethod
    def _lin_init_one(m):
        if isinstance(m, nn.Linear):
            # weight
            m.weight = torch.nn.Parameter(torch.ones(m.weight.shape))
            # bias
            m.bias = torch.nn.Parameter(torch.zeros(m.bias.shape))

    @property
    def output_dim(self):
        return self.out_dim


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> List[str]:
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[B, T, Num_labels]`.

        Returns:
          List[str]: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        btokens = []
        lens = []
        for b in range(indices.shape[0]):
            tokens = []
            for k in range(indices.shape[1]):
                t = indices[b, k]
                if t != self.blank:
                    tokens.append(t)
            lens.append(len(tokens))
            btokens.append(torch.IntTensor(tokens))
        btokens = collate_vectors(btokens, padding_value=self.blank)
        lens = torch.IntTensor(lens)
        return btokens, lens

    def idx2labels(self, indices):
        joined = "".join([self.labels[i] for i in indices])
        return joined.replace("|", " ").strip().split()


class WarmupStepLR(torch.optim.lr_scheduler._LRScheduler):
    """Linear Decay rate scheduler with warm up."""

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_updates: int,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.warmup_updates = warmup_updates
        super().__init__(
            optimizer,
            step_size=step_size,
            gammar=gamma,
            last_epoch=last_epoch,
            verbose=verbose,
        )

    def get_lr(self):
        if self._step_count <= self.warmup_updates:
            return [
                self._step_count / self.warmup_updates * base_lr
                for base_lr in self.base_lrs
            ]
        else:
            return super().get_lr()


class WarmupLR(torch.optim.lr_scheduler._LRScheduler):
    """The WarmupLR scheduler

    This scheduler is almost same as NoamLR Scheduler except for following difference:

    NoamLR:
        lr = optimizer.lr * model_size ** -0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)
    WarmupLR:
        lr = optimizer.lr * warmup_step ** 0.5
             * min(step ** -0.5, step * warmup_step ** -1.5)

    Note that the maximum lr equals to optimizer.lr in this scheduler.

    Based on Apache 2.0
    https://espnet.github.io/espnet/_modules/espnet2/schedulers/warmup_lr.html#WarmupLR
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 25000,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = self.last_epoch + 1
        return [
            lr
            * self.warmup_steps**0.5
            * min(step_num**-0.5, step_num * self.warmup_steps**-1.5)
            for lr in self.base_lrs
        ]
