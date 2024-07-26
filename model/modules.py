import os
import json
import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.tools import get_mask_from_lengths, pad
from .submodels import VariantDurationPredictor, TemporalVariancePredictor, DynamicDurationPredictor, NormalizedEmbedding

from typing import Optional, Tuple
from numba import jit, prange

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


@jit(nopython=True)
def mas_width1(attn_map):
    """mas with hardcoded width=1"""
    # assumes mel x text
    opt = np.zeros_like(attn_map)
    attn_map = np.log(attn_map)
    attn_map[0, 1:] = -np.inf
    log_p = np.zeros_like(attn_map)
    log_p[0, :] = attn_map[0, :]
    prev_ind = np.zeros_like(attn_map, dtype=np.int64)
    for i in range(1, attn_map.shape[0]):
        for j in range(attn_map.shape[1]):  # for each text dim
            prev_log = log_p[i - 1, j]
            prev_j = j

            if j - 1 >= 0 and log_p[i - 1, j - 1] >= log_p[i - 1, j]:
                prev_log = log_p[i - 1, j - 1]
                prev_j = j - 1

            log_p[i, j] = attn_map[i, j] + prev_log
            prev_ind[i, j] = prev_j

    # now backtrack
    curr_text_idx = attn_map.shape[1] - 1
    for i in range(attn_map.shape[0] - 1, -1, -1):
        opt[i, curr_text_idx] = 1
        curr_text_idx = prev_ind[i, curr_text_idx]
    opt[0, curr_text_idx] = 1
    return opt


@jit(nopython=True)
def b_mas(b_attn_map, in_lens, out_lens, width=1):
    assert width == 1
    attn_out = np.zeros_like(b_attn_map)

    for b in prange(b_attn_map.shape[0]):
        out = mas_width1(b_attn_map[b, 0, : out_lens[b], : in_lens[b]])
        attn_out[b, 0, : out_lens[b], : in_lens[b]] = out
    return attn_out


def binarize_attention_parallel(attn, in_lens, out_lens):
    """For training purposes only. Binarizes attention with MAS.
           These will no longer receive a gradient.
        Args:
            attn: B x 1 x max_mel_len x max_text_len
        """
    with torch.no_grad():
        attn_cpu = attn.data.cpu().numpy()
        attn_out = b_mas(attn_cpu, in_lens.cpu().numpy(), out_lens.cpu().numpy(), width=1)
    return torch.from_numpy(attn_out).to(attn.device)


class PartialConv1d(torch.nn.Conv1d):
    """
    Zero padding creates a unique identifier for where the edge of the data is, such that the model can almost always identify
    exactly where it is relative to either edge given a sufficient receptive field. Partial padding goes to some lengths to remove
    this affect.
    """

    def __init__(self, *args, **kwargs):
        super(PartialConv1d, self).__init__(*args, **kwargs)
        weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0])
        self.register_buffer("weight_maskUpdater", weight_maskUpdater, persistent=False)
        slide_winsize = torch.tensor(self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2])
        self.register_buffer("slide_winsize", slide_winsize, persistent=False)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1)
            self.register_buffer('bias_view', bias_view, persistent=False)
        # caching part
        self.last_size = (-1, -1, -1)

        update_mask = torch.ones(1, 1, 1)
        self.register_buffer('update_mask', update_mask, persistent=False)
        mask_ratio = torch.ones(1, 1, 1)
        self.register_buffer('mask_ratio', mask_ratio, persistent=False)
        self.partial: bool = True

    def calculate_mask(self, input: torch.Tensor, mask_in: Optional[torch.Tensor]):
        with torch.no_grad():
            if mask_in is None:
                mask = torch.ones(1, 1, input.shape[2], dtype=input.dtype, device=input.device)
            else:
                mask = mask_in
            update_mask = F.conv1d(
                mask,
                self.weight_maskUpdater,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=1,
            )
            # for mixed precision training, change 1e-8 to 1e-6
            mask_ratio = self.slide_winsize / (update_mask + 1e-6)
            update_mask = torch.clamp(update_mask, 0, 1)
            mask_ratio = torch.mul(mask_ratio.to(update_mask), update_mask)
            return torch.mul(input, mask), mask_ratio, update_mask

    def forward_aux(self, input: torch.Tensor, mask_ratio: torch.Tensor, update_mask: torch.Tensor) -> torch.Tensor:
        assert len(input.shape) == 3

        raw_out = self._conv_forward(input, self.weight, self.bias)

        if self.bias is not None:
            output = torch.mul(raw_out - self.bias_view, mask_ratio) + self.bias_view
            output = torch.mul(output, update_mask)
        else:
            output = torch.mul(raw_out, mask_ratio)

        return output

    @torch.jit.ignore
    def forward_with_cache(self, input: torch.Tensor, mask_in: Optional[torch.Tensor] = None) -> torch.Tensor:
        use_cache = not (torch.jit.is_tracing() or torch.onnx.is_in_onnx_export())
        cache_hit = use_cache and mask_in is None and self.last_size == input.shape
        if cache_hit:
            mask_ratio = self.mask_ratio
            update_mask = self.update_mask
        else:
            input, mask_ratio, update_mask = self.calculate_mask(input, mask_in)
            if use_cache:
                # if a mask is input, or tensor shape changed, update mask ratio
                self.last_size = tuple(input.shape)
                self.update_mask = update_mask
                self.mask_ratio = mask_ratio
        return self.forward_aux(input, mask_ratio, update_mask)

    def forward_no_cache(self, input: torch.Tensor, mask_in: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.partial:
            input, mask_ratio, update_mask = self.calculate_mask(input, mask_in)
            return self.forward_aux(input, mask_ratio, update_mask)
        else:
            if mask_in is not None:
                input = torch.mul(input, mask_in)
            return self._conv_forward(input, self.weight, self.bias)

    def forward(self, input: torch.Tensor, mask_in: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.partial:
            return self.forward_with_cache(input, mask_in)
        else:
            if mask_in is not None:
                input = torch.mul(input, mask_in)
            return self._conv_forward(input, self.weight, self.bias)


class ConvNorm(torch.nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=None,
            dilation=1,
            bias=True,
            w_init_gain='linear',
            use_partial_padding: bool = False,
            use_weight_norm: bool = False,
            norm_fn=None,
    ):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)
        self.use_partial_padding: bool = use_partial_padding
        conv = PartialConv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        conv.partial = use_partial_padding
        torch.nn.init.xavier_uniform_(conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
        if use_weight_norm:
            conv = torch.nn.utils.weight_norm(conv)
        if norm_fn is not None:
            self.norm = norm_fn(out_channels, affine=True)
        else:
            self.norm = None
        self.conv = conv

    def forward(self, input: torch.Tensor, mask_in: Optional[torch.Tensor] = None) -> torch.Tensor:
        ret = self.conv(input, mask_in)
        if self.norm is not None:
            ret = self.norm(ret)
        return ret


class SafeSoftmax(nn.Module):
    def __init__(self, dim=-1):
        super(SafeSoftmax, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Handle inf by replacing them with very large finite numbers
        large_value = torch.finfo(x.dtype).max
        x_replaced = torch.where(x == float('inf'), torch.tensor(large_value, dtype=x.dtype, device=x.device), x)
        x_replaced = torch.where(x == float('-inf'), torch.tensor(-large_value, dtype=x.dtype, device=x.device),
                                 x_replaced)

        # Subtract the max value for numerical stability
        x_stable = x_replaced - torch.max(x_replaced, dim=self.dim, keepdim=True).values
        return F.softmax(x_stable, self.dim)


class SafeLogSoftmax(nn.Module):
    def __init__(self, dim=-1):
        super(SafeLogSoftmax, self).__init__()
        self.dim = dim

    def forward(self, x):
        # Handle inf by replacing them with very large finite numbers
        large_value = torch.finfo(x.dtype).max
        x_replaced = torch.where(x == float('inf'), torch.tensor(large_value, dtype=x.dtype, device=x.device), x)
        x_replaced = torch.where(x == float('-inf'), torch.tensor(-large_value, dtype=x.dtype, device=x.device),
                                 x_replaced)

        # Subtract the max value for numerical stability
        x_stable = x_replaced - torch.max(x_replaced, dim=self.dim, keepdim=True).values
        return F.log_softmax(x_stable, self.dim)


class AlignmentEncoder(torch.nn.Module):
    """Module for alignment text and mel spectrogram. """

    def __init__(
            self, n_mel_channels=80, n_text_channels=512, n_att_channels=80, temperature=0.0005,
    ):
        super().__init__()
        self.temperature = temperature
        self.softmax = SafeSoftmax(dim=3)
        self.log_softmax = SafeLogSoftmax(dim=3)

        self.key_proj = nn.Sequential(
            ConvNorm(n_text_channels, n_text_channels * 2, kernel_size=3, bias=True, w_init_gain='relu'),
            torch.nn.ReLU(),
            ConvNorm(n_text_channels * 2, n_att_channels, kernel_size=1, bias=True),
        )

        self.query_proj = nn.Sequential(
            ConvNorm(n_mel_channels, n_mel_channels * 2, kernel_size=3, bias=True, w_init_gain='relu'),
            torch.nn.ReLU(),
            ConvNorm(n_mel_channels * 2, n_mel_channels, kernel_size=1, bias=True),
            torch.nn.ReLU(),
            ConvNorm(n_mel_channels, n_att_channels, kernel_size=1, bias=True),
        )

    def get_dist(self, keys, queries, mask=None):
        """Calculation of distance matrix.
        Args:
            queries (torch.tensor): B x C x T1 tensor (probably going to be mel data).
            keys (torch.tensor): B x C2 x T2 tensor (text data).
            mask (torch.tensor): B x T2 x 1 tensor, binary mask for variable length entries and also can be used
                for ignoring unnecessary elements from keys in the resulting distance matrix (True = mask element, False = leave unchanged).
        Output:
            dist (torch.tensor): B x T1 x T2 tensor.
        """
        keys_enc = self.key_proj(keys)  # B x n_attn_dims x T2
        queries_enc = self.query_proj(queries)  # B x n_attn_dims x T1
        attn = (queries_enc[:, :, :, None] - keys_enc[:, :, None]) ** 2  # B x n_attn_dims x T1 x T2
        dist = attn.sum(1, keepdim=True)  # B x 1 x T1 x T2

        if mask is not None:
            dist.data.masked_fill_(mask.permute(0, 2, 1).unsqueeze(2), float('-1e4'))

        return dist.squeeze(1)

    @staticmethod
    def get_durations(attn_soft, text_len, spect_len):
        """Calculation of durations.
        Args:
            attn_soft (torch.tensor): B x 1 x T1 x T2 tensor.
            text_len (torch.tensor): B tensor, lengths of text.
            spect_len (torch.tensor): B tensor, lengths of mel spectrogram.
        """
        attn_hard = binarize_attention_parallel(attn_soft, text_len, spect_len)
        durations = attn_hard.sum(2)[:, 0, :]
        assert torch.all(torch.eq(durations.sum(dim=1), spect_len))
        return durations

    @staticmethod
    def get_mean_dist_by_durations(dist, durations, mask=None):
        """Select elements from the distance matrix for the given durations and mask and return mean distance.
        Args:
            dist (torch.tensor): B x T1 x T2 tensor.
            durations (torch.tensor): B x T2 tensor. Dim T2 should sum to T1.
            mask (torch.tensor): B x T2 x 1 binary mask for variable length entries and also can be used
                for ignoring unnecessary elements in dist by T2 dim (True = mask element, False = leave unchanged).
        Output:
            mean_dist (torch.tensor): B x 1 tensor.
        """
        batch_size, t1_size, t2_size = dist.size()
        assert torch.all(torch.eq(durations.sum(dim=1), t1_size))

        if mask is not None:
            dist = dist.masked_fill(mask.permute(0, 2, 1).unsqueeze(2), 0)

        # TODO(oktai15): make it more efficient
        mean_dist_by_durations = []
        for dist_idx in range(batch_size):
            mean_dist_by_durations.append(
                torch.mean(
                    dist[
                        dist_idx,
                        torch.arange(t1_size),
                        torch.repeat_interleave(torch.arange(t2_size), repeats=durations[dist_idx]),
                    ]
                )
            )

        return torch.tensor(mean_dist_by_durations, dtype=dist.dtype, device=dist.device)

    @staticmethod
    def get_mean_distance_for_word(l2_dists, durs, start_token, num_tokens):
        """Calculates the mean distance between text and audio embeddings given a range of text tokens.
        Args:
            l2_dists (torch.tensor): L2 distance matrix from Aligner inference. T1 x T2 tensor.
            durs (torch.tensor): List of durations corresponding to each text token. T2 tensor. Should sum to T1.
            start_token (int): Index of the starting token for the word of interest.
            num_tokens (int): Length (in tokens) of the word of interest.
        Output:
            mean_dist_for_word (float): Mean embedding distance between the word indicated and its predicted audio frames.
        """
        # Need to calculate which audio frame we start on by summing all durations up to the start token's duration
        start_frame = torch.sum(durs[:start_token]).data

        total_frames = 0
        dist_sum = 0

        # Loop through each text token
        for token_ind in range(start_token, start_token + num_tokens):
            # Loop through each frame for the given text token
            for frame_ind in range(start_frame, start_frame + durs[token_ind]):
                # Recall that the L2 distance matrix is shape [spec_len, text_len]
                dist_sum += l2_dists[frame_ind, token_ind]

            # Update total frames so far & the starting frame for the next token
            total_frames += durs[token_ind]
            start_frame += durs[token_ind]

        return dist_sum / total_frames

    def forward(self, queries, keys, mask=None, attn_prior=None, conditioning=None):
        """Forward pass of the aligner encoder.
        Args:
            queries (torch.tensor): B x C x T1 tensor (probably going to be mel data).
            keys (torch.tensor): B x C2 x T2 tensor (text data).
            mask (torch.tensor): B x T2 x 1 tensor, binary mask for variable length entries (True = mask element, False = leave unchanged).
            attn_prior (torch.tensor): prior for attention matrix.
            conditioning (torch.tensor): B x T2 x 1 conditioning embedding
        Output:
            attn (torch.tensor): B x 1 x T1 x T2 attention mask. Final dim T2 should sum to 1.
            attn_logprob (torch.tensor): B x 1 x T1 x T2 log-prob attention mask.
        """
        if conditioning is not None:
            keys = keys + conditioning.transpose(1, 2)

        keys_enc = self.key_proj(keys)  # B x n_attn_dims x T2
        queries_enc = self.query_proj(queries)  # B x n_attn_dims x T1

        # Simplistic Gaussian Isotopic Attention
        attn = (queries_enc[:, :, :, None] - keys_enc[:, :, None]) ** 2  # B x n_attn_dims x T1 x T2
        attn = -self.temperature * attn.sum(1, keepdim=True)

        if attn_prior is not None:
            attn = self.log_softmax(attn) + torch.log(attn_prior[:, None] + 1e-8)

        attn_logprob = attn.clone()

        if mask is not None:  # Original aligner uses -inf for the masked fill, but here we use -1e4 for stability during fp16 training.
            attn.data.masked_fill_(mask.permute(0, 2, 1).unsqueeze(2), float('-1e4'))

        attn = self.softmax(attn)  # softmax along T2
        return attn, attn_logprob


class TransposeBatchNorm(nn.Module):
    def __init__(self, num_features):
        super(TransposeBatchNorm, self).__init__()
        self.batch_norm = nn.BatchNorm1d(num_features)

    def forward(self, x):
        # Transpose to (batch_size, num_features, sequence_length)
        x = x.transpose(1, 2)
        # Apply BatchNorm1d
        x = self.batch_norm(x)
        # Transpose back to (batch_size, sequence_length, num_features)
        x = x.transpose(1, 2)
        return x


class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()

        dp_type = model_config["duration_predictor"]["type"]

        if dp_type == "tcn":
            self.duration_predictor = DynamicDurationPredictor(
                num_inputs=model_config["transformer"]["encoder_hidden"],
                num_channels=model_config["duration_predictor"]["tcn_channels"],
                kernel_sizes=model_config["duration_predictor"]["tcn_kernel_sizes"],
                dropout=model_config["duration_predictor"]["dropout"],
                start_i=1,
                att_dropout=model_config["duration_predictor"]["att_dropout"],
                heads=model_config["duration_predictor"]["tcn_heads"],
                bidirectional=model_config["duration_predictor"]["bidirectional"],
                backwards_channels=model_config["duration_predictor"]["backwards_tcn_channels"],
                backwards_heads=model_config["duration_predictor"]["backwards_heads"],
                backwards_kernel_sizes=model_config["duration_predictor"]["backwards_kernel_sizes"],
            )
            dp_output_channels = model_config["duration_predictor"]["tcn_channels"][-1]
        elif dp_type == "lstm":
            self.duration_predictor = VariantDurationPredictor(
                text_channels=model_config["transformer"]["encoder_hidden"],
                filter_channels=model_config["duration_predictor"]["filter_size"],
                depth=model_config["duration_predictor"]["decoder_depth"],
                heads=model_config["duration_predictor"]["heads"],
                p_dropout=model_config["duration_predictor"]["dropout"],
                final_dropout=model_config["duration_predictor"]["att_dropout"],
                kernel_size=model_config["duration_predictor"]["kernel_size"],
                conv_depth=model_config["duration_predictor"]["conv_depth"],
                start_i=3,
                lstm_bidirectional=model_config["duration_predictor"]["bidirectional"],
            )
            dp_output_channels = model_config["duration_predictor"]["filter_size"]
        else:
            raise RuntimeError(f"Invalid duration predictor type: {dp_type}. Valid are tcn and lstm")

        self.length_regulator = LengthRegulator()
        self.pitch_predictor = TemporalVariancePredictor(
            input_channels=model_config["transformer"]["encoder_hidden"],
            num_channels=model_config["variance_predictor"]["filter_size"],
            kernel_size=model_config["variance_predictor"]["kernel_size"],
            dropout=model_config["variance_predictor"]["dropout"],
            cond_input_size=dp_output_channels,
        )
        self.energy_predictor = TemporalVariancePredictor(
            input_channels=model_config["transformer"]["encoder_hidden"],
            num_channels=model_config["variance_predictor"]["filter_size"],
            kernel_size=model_config["variance_predictor"]["kernel_size"],
            dropout=model_config["variance_predictor"]["dropout"],
            cond_input_size=dp_output_channels,
        )

        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]
        with open(
                os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]
            energy_min, energy_max = stats["energy"][:2]

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )

        # I'll make these into NormalizedEmbeddings later
        self.pitch_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )
        # BatchNorm is better than LayerNorm at regularization, our main objective here
        self.pitch_norm = TransposeBatchNorm(model_config["transformer"]["encoder_hidden"])
        self.energy_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )
        self.energy_norm = TransposeBatchNorm(model_config["transformer"]["encoder_hidden"])

        self.pitch_energy_drop = nn.Dropout(model_config["variance_predictor"]["dropout_on_emb"])

    def get_pitch_embedding(self, x, target, mask, control, y, y_mask):
        prediction = self.pitch_predictor(x, mask, y, y_mask)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            prediction = prediction * control
            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )

        embedding = self.pitch_norm(embedding)
        embedding = self.pitch_energy_drop(embedding)
        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control, y, y_mask):
        prediction = self.energy_predictor(x, mask, y, y_mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            prediction = prediction * control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )

        embedding = self.energy_norm(embedding)
        embedding = self.pitch_energy_drop(embedding)
        return prediction, embedding

    def forward(
            self,
            x,
            src_mask,
            src_lens,
            mel_mask=None,
            max_len=None,
            pitch_target=None,
            energy_target=None,
            duration_target=None,
            p_control=1.0,
            e_control=1.0,
            d_control=1.0,
    ):
        log_duration_prediction, x_mask, dur_hidden = self.duration_predictor(x, src_lens)


        if self.pitch_feature_level == "phoneme_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, src_mask, p_control, dur_hidden, x_mask
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "phoneme_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, src_mask, p_control, dur_hidden, x_mask
            )
            x = x + energy_embedding

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = get_mask_from_lengths(mel_len)

        if self.pitch_feature_level == "frame_level":
            pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                x, pitch_target, mel_mask, p_control, dur_hidden, x_mask
            )
            x = x + pitch_embedding
        if self.energy_feature_level == "frame_level":
            energy_prediction, energy_embedding = self.get_energy_embedding(
                x, energy_target, mel_mask, p_control, dur_hidden, x_mask
            )
            x = x + energy_embedding

        return (
            x,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
        )


class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len):
        output = list()
        mel_len = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, batch, predicted):
        out = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, max_len):
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
            w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x

