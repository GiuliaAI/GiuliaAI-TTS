import torch
import torch.nn as nn
from .attentions import TransformerEncoder, TransformerDecoder, TemporalConvNet, TCNAttention, MultiHeadAttention
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F


# Applying LayerNorm + Dropout on embeddings increases performance, probably due to the regularizing effect
# Thanks dathudeptrai from TensorFlowTTS for discovering this.
class NormalizedEmbedding(nn.Module):
    """
    Embedding + LayerNorm + Dropout
    """

    def __init__(self, num_embeddings, embedding_dim, dropout=0.1):
        super(NormalizedEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.embedding(x)
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


class StochasticDropout(nn.Module):
    """
    Also known as Monte Carlo dropout, StochasticDropout keeps a lower dropout rate to apply during inference, for stochasticity.
    If not specified, it will be dropout / 3, with the minimum being 0.1
    """

    def __init__(self, p=0.5, p_inference=None, min_p_inference=0.1, stochastic=False):
        super(StochasticDropout, self).__init__()
        self.p = p  # Dropout probability during training

        if p_inference is None:
            p_inference = max(min_p_inference, p / 3)  # ensure stochastic dropout is at least 0.1

        self.p_inference = p_inference  # Dropout probability during inference
        self.stochastic = stochastic

    def forward(self, x):
        if self.training:
            return F.dropout(x, self.p, self.training)
        else:
            if self.stochastic:
                return F.dropout(x, self.p, True)
            else:
                return F.dropout(x, self.p, self.training)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, forward_expansion, dropout, alibi_alpha=1.0,
                 start_i=0):
        super(TextEncoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.emb_norm = nn.LayerNorm(embed_size)
        self.encoder = TransformerEncoder(embed_size, num_heads, num_layers, forward_expansion, dropout,
                                          alibi_alpha=alibi_alpha, start_i=start_i)
        self.dropout = StochasticDropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_size)

    def forward(self, token_ids, seq_lens):
        # Embed token_ids
        x = self.embed(token_ids)  # Shape: (batch, max_seq_len, embed_size)
        x = self.emb_norm(x)
        x = self.dropout(x)

        # Create a mask based on sequence lengths
        max_len = token_ids.size(1)
        mask = torch.arange(max_len, device=seq_lens.device).expand(len(seq_lens), max_len) >= seq_lens.unsqueeze(1)

        # Pass through the transformer encoder
        x = self.encoder(x, mask.unsqueeze(1).unsqueeze(2))

        # Apply dropout and LayerNorm after the encoder
        x = self.dropout(x)
        x = self.layer_norm(x)

        return x


class SConvNorm(nn.Module):
    def __init__(self, channels, num_layers=2, kernel_size=3, p_dropout=0.1):
        super(SConvNorm, self).__init__()
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(channels),
                nn.Dropout(p_dropout),
                nn.ReLU()
            ) for _ in range(num_layers)
        ])
        # x = tensor size (batch, channels, seq_len)

    def forward(self, x):
        for layer in self.conv_layers:
            residual = x
            x = layer(x)
            x = x + residual  # Adding residual connection
        return x


def lens_to_sequence_mask(lens_unpacked, max_seq_len):
    """
    Create a sequence mask for padded sequences.

    Args:
    lens_unpacked (Tensor): A 1D tensor containing the lengths of each sequence in the batch.
    max_seq_len (int): The maximum sequence length in the batch.

    Returns:
    Tensor: A binary mask of size (batch, 1, seq_len) where 1 indicates valid positions and 0 indicates padding.
    """
    batch_size = lens_unpacked.size(0)
    mask = torch.arange(max_seq_len).to(lens_unpacked.device).expand(batch_size, max_seq_len) < lens_unpacked.unsqueeze(
        1)
    mask = mask.unsqueeze(1).float()  # Add channel dimension and convert to float
    return mask.to(lens_unpacked.device)


def pad_to_original_length(lstm_output, original_seq_len, current_seq_len):
    """
    Pad the LSTM output back to the original input sequence length.

    Args:
    lstm_output (Tensor): The output tensor from the LSTM.
    original_seq_len (int): The original sequence length before any padding was applied.
    current_seq_len (int): The current sequence length after the LSTM processing.

    Returns:
    Tensor: Padded LSTM output tensor.
    """
    padding_length = original_seq_len - current_seq_len
    if padding_length > 0:
        # Pad the sequences to match the original input length
        padded_output = F.pad(lstm_output, [0, 0, 0, padding_length])
    else:
        padded_output = lstm_output

    return padded_output


def generate_masks_from_float_mask(float_mask):
    # Ensure src_mask is in the correct format for the Transformer which expects (batch, 1, 1, seq_len)
    src_mask = float_mask.bool().unsqueeze(1)  # Shape: (batch, 1, 1, seq_len)

    # Generate target mask combining src_mask with subsequent mask for causal attention
    seq_len = float_mask.size(2)
    subsequent_mask = torch.tril(torch.ones((seq_len, seq_len), device=float_mask.device, dtype=torch.bool))
    tgt_mask = src_mask & subsequent_mask.unsqueeze(0)  # Expand subsequent_mask to batch size

    return src_mask, tgt_mask


class VariantDurationPredictor(nn.Module):
    def __init__(self, text_channels, filter_channels=512, depth=4, heads=4, kernel_size=3, p_dropout=0.2,
                 final_dropout=0.2, conv_depth=2, lstm_bidirectional=True, start_i=0):
        super(VariantDurationPredictor, self).__init__()

        print("Using Variant Duration Predictor")
        self.use_dual_proj = False
        self.lstm_bidirectional = lstm_bidirectional

        if conv_depth > 0:
            self.pre_convs = SConvNorm(filter_channels, num_layers=conv_depth, kernel_size=kernel_size)
            self.post_conv_drop = StochasticDropout(p_dropout)
        else:
            print("Not using pre convs")
            self.pre_convs = nn.Identity()
            self.post_conv_drop = nn.Identity()

        self.decoder = TransformerDecoder(embed_size=filter_channels, heads=heads, num_layers=depth,
                                          forward_expansion=4, dropout=p_dropout, alibi_alpha=1.5, mode="conv",
                                          kernel_size=3, start_i=start_i)

        lstm_channels = filter_channels // 2

        self.lstm_channels = lstm_channels

        self.lstm = nn.LSTM(input_size=filter_channels, hidden_size=lstm_channels, batch_first=True,
                            bidirectional=self.lstm_bidirectional)
        if self.lstm_bidirectional:
            print("BiLSTM")
            self.fc_merge = nn.Linear(2 * self.lstm_channels,
                                      self.lstm_channels)  # Merging down to the original filter_channels size

        self.out_proj = nn.Conv1d(in_channels=lstm_channels, out_channels=1, kernel_size=1)

        self.final_dropout = StochasticDropout(final_dropout)
        self.use_pre_proj = False

        if text_channels != filter_channels:
            self.pre_proj = nn.Conv1d(text_channels, filter_channels, 1)
            self.use_pre_proj = True

    # x = Encoder hidden states size (batch, seq_len, text_channels)
    # x_lengths = Lengths of x size (batch_size,)
    def forward(self, x, x_lengths):
        x = x.transpose(1, 2)  # (batch, seq_len, text_channels) => (batch, text_channels, seq_len)

        x_mask = lens_to_sequence_mask(x_lengths, x.size(2))
        if self.use_pre_proj:
            x = self.pre_proj(x)

        x = self.pre_convs(x)  # x = (b, channels, seq_len)
        x = self.post_conv_drop(x)

        # Apply mask after convolutions
        x = x * x_mask

        # Perform transformer stuff

        # Transpose for decoder
        x = x.transpose(1, 2)  # (b, text_channels, seq_len) -> (b, seq_len, channels)

        src_mask, tgt_mask = generate_masks_from_float_mask(x_mask)

        # Transformer pass

        x_dec = self.decoder(x, x, src_mask,
                             tgt_mask)  # x enc = (b, seq_len, channels)

        # Residual connection.
        x = x + x_dec
        # Apply dropout even in inference
        x = self.final_dropout(x)

        # LSTM Portion
        # input must be (batch, seq_len, channels)

        # Pack padded sequence

        # max(x_lengths) is less than the seq_len dimension in x, often by 5 or 10
        # and the LSTM outputs a tensor of max(x_lengths) in its seq_len dimension
        # therefore, we save the original seq_len dimension for padding back later
        x_seq_len_orig = x.size(1)

        x = pack_padded_sequence(x, x_lengths.detach().cpu(),
                                 # pack_padded_sequence demands that the lengths tensor be on the CPU
                                 batch_first=True, enforce_sorted=False)

        # LSTM pass
        x, (hn, cn) = self.lstm(x)

        # Unpack the sequence
        x, lens_unpacked = pad_packed_sequence(x, batch_first=True)  # x_lstm:  (batch, seq_len, lstm_channels)
        x = self.final_dropout(x)
        # pad back to pre-LSTM seq_len
        x = pad_to_original_length(x, x_seq_len_orig, x.size(1))

        if self.lstm_bidirectional:
            x = self.fc_merge(x)

        # Transpose dimensions for the post-convolution
        x = x.transpose(1, 2)  # (b, seq_len, channels) -> (b, channels, seq_len)
        x = self.final_dropout(x)

        # Project using 1D convolution
        log_durations = self.out_proj(x)

        log_durations *= x_mask

        log_durations = log_durations.squeeze(1)  # (batch, 1, seq_len) => (batch, seq_len)

        x_mask = x_mask.squeeze(1).bool()
        x_mask = ~x_mask

        return log_durations, x_mask, x


# VariancePredictor but using TCNs for cheaper-than-RNN temporal dependencies.
class TemporalVariancePredictor(nn.Module):

    def __init__(self, input_channels, num_channels, kernel_size=2, dropout=0.2, cond_input_size=None):
        super(TemporalVariancePredictor, self).__init__()
        # Temporal Convolutional Network
        # Multiplicative dilation growth gives the best balance between coverage and accuracy.
        self.tcn = TemporalConvNet(input_channels, num_channels, kernel_size=kernel_size, dropout=dropout,
                                   dilation_growth="mul", use_se=True)
        self.final_drop = StochasticDropout(dropout)
        self.cond_input_size = cond_input_size
        self.input_channels = input_channels

        self.output_layer = nn.Linear(num_channels[-1], 1)

        # Duration-Spectrogram pre-conditioning
        # Conv1D => ReLU => AllAttention(duration_hidden,spec) => LayerNorm => ReLU => Dropout
        if self.cond_input_size is not None:

            self.cond_proj = nn.Conv1d(self.cond_input_size, self.input_channels,
                                       3, padding="same")

            # For some reason, this case benefits from having 4 heads instead of 2
            self.cond_attention = MultiHeadAttention(self.input_channels, 4, alibi_alpha=1.5,
                                                     start_i_increment=4, num_persistent=16)

            self.cond_norm = nn.LayerNorm(self.input_channels)

            self.cond_act = nn.ReLU()
            self.inter_cond_drop = nn.Dropout(0.25)
            self.cond_drop = nn.Dropout(0.5)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def make_cond_vector(self, x, y, x_mask, y_mask):
        """
        Create conditioning vector
        :param x: Input size (batch, seq_len, channels)
        :param y: Conditioning size  (batch, cond_len, cond_channels)
        :param x_mask: Mask of x size (batch, seq_len), where True is padded
        :param y_mask: Mask of y size (batch, cond_len), where True is padded

        :return: Conditioning vector, ready to add to hidden states
        """
        x_mask_expanded = x_mask.unsqueeze(1).unsqueeze(3)  # Shape: (batch_size, 1, mel_len, 1)
        y_mask_expanded = y_mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, duration_len)

        # Combine masks using broadcasting
        attention_mask = x_mask_expanded & y_mask_expanded  # Shape: (batch_size, 1, mel_len, duration_len)
        attention_mask = ~attention_mask  # True=padded => True=valid

        # (batch, seq_len, channels) <<=> (batch, channels, seq_len)
        y = self.cond_proj(
            y.transpose(1, 2)
        ).transpose(1,2)

        y = self.cond_act(y)
        y = self.inter_cond_drop(y)

        z = self.cond_attention(y, y, x, mask=attention_mask)
        z = self.inter_cond_drop(z)

        z = self.cond_norm(z)
        z = self.cond_act(z)

        return z

    def forward(self, x, x_mask, y, y_mask):
        """
        Forward pass through the Temporal Variance Predictor

        :param x: Input hidden size (batch, seq_len, channels)
        :param x_mask: Mask of x size (batch, seq_len), where True is padded
        :param y: Conditioning hidden size (batch, cond_len, cond_channels)
        :param y_mask: Mask of y size (batch, cond_len), where True is padded

        :return: Predictions
        """

        if self.cond_input_size is not None:
            # Prevent backpropagation into the duration predictor
            # If we don't do this, it tries to optimize the DP for pitch&energy and severely overfits
            y = y.detach()

            cond = self.make_cond_vector(x, y, x_mask, y_mask)
            cond = self.cond_drop(cond)
            x = x + cond

        # Pass input through the TCN
        x = x.transpose(1, 2)  # (batch, seq_len, channels) => (batch, channels, seq_len)
        x_mask = x_mask.unsqueeze(1)  # (batch, seq_len) => (batch, 1, seq_len)

        x = self.tcn(x)  # x = (batch, channels, seq_len)

        if x_mask is not None:
            x = x.masked_fill(x_mask, 0.0)

        # linear pass
        x = x.transpose(1, 2)  # x = (batch, seq_len, channels)

        x = self.final_drop(x)

        # output
        x = self.output_layer(x)  # x = (batch, seq_len, 1)

        # squeeze 4 output
        x = x.squeeze(-1)  # (batch, seq_len, 1) => (batch, seq_len)

        if x_mask is not None:  # x_mask: (batch, 1, seq_len) => (batch, seq_len)
            x = x.masked_fill(x_mask.squeeze(1), 0.0)

        return x


class SpectrogramDecoder(nn.Module):
    def __init__(self, input_size, filter_channels, mel_channels, depth, heads, kernel_sizes, dropout=0.1,
                 alibi_alpha=1.0, forward_expansion=3):
        super(SpectrogramDecoder, self).__init__()

        self.input_size = input_size
        self.filter_channels = filter_channels

        if input_size != filter_channels:
            self.pre_fc = nn.Linear(input_size, filter_channels)

        self.dec = TransformerDecoder(filter_channels,
                                      heads=heads, num_layers=depth,
                                      forward_expansion=forward_expansion, dropout=dropout, alibi_alpha=alibi_alpha,
                                      mode="conv", kernel_sizes=kernel_sizes)

        self.mel_fc = nn.Linear(filter_channels, mel_channels)

    # x_mask : True=exclude mask size (batch, mel_lens)
    # x: (batch, mel_lens, channels)
    def forward(self, x, x_mask):

        orig_mask = x_mask.clone()

        # (batch, mel_lens) -> (batch, 1, mel_lens)
        x_mask = x_mask.unsqueeze(1)
        # True=exclude => True=Include
        x_mask = ~x_mask

        src_mask, tgt_mask = generate_masks_from_float_mask(x_mask)

        # Decoder pass

        if self.input_size != self.filter_channels:
            x = self.pre_fc(x)

        x = self.dec(x, x, src_mask,
                     tgt_mask)

        x = self.mel_fc(x)

        return x, orig_mask


def mask_to_attention_mask(mask):
    attention_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
    attention_mask = attention_mask.unsqueeze(1)
    # Flip the mask, our attention uses True=valid
    attention_mask = ~attention_mask
    return attention_mask


class DynamicDurationPredictor(nn.Module):
    """
    DynamicDurationPredictor:

    (optionally bidirectional) TCN-Attention with increasing kernel sizes => Projection
    """
    def __init__(self, num_inputs, num_channels, kernel_sizes=[2, 2, 3], dropout=0.2, att_dropout=0.3, alibi_alpha=1.5,
                 start_i=0, heads=2, bidirectional=False, backwards_channels=[256, 256], backwards_heads=[2,2],
                 backwards_kernel_sizes=[2, 3]):
        super(DynamicDurationPredictor, self).__init__()

        self.tcn_output_channels = num_channels[-1]
        self.bidirectional = bidirectional

        # Initialize the TCNAttention module
        self.tcn_attention = TCNAttention(num_inputs, num_channels, kernel_sizes, dropout, att_dropout, heads,
                                          alibi_alpha=alibi_alpha, start_i_increment=start_i)
        if self.bidirectional:
            self.backwards_tcn_attention = TCNAttention(num_inputs, backwards_channels, backwards_kernel_sizes, dropout, att_dropout,
                                                        backwards_heads,
                                                        alibi_alpha=alibi_alpha, start_i_increment=start_i)

            self.bw_tcn_output_channels = backwards_channels[-1]

            # prevent model from overrelying on backwards features
            self.backwards_drop = nn.Dropout(att_dropout)
            self.fw_projection = nn.Linear(self.tcn_output_channels + self.bw_tcn_output_channels, self.tcn_output_channels)

        self.final_drop = nn.Dropout(0.1)
        self.linear_projection = nn.Linear(self.tcn_output_channels, 1)

    def forward(self, x, x_lengths):
        """
        Forward pass through the DynamicDurationPredictor

        :param x: Encoded text size (batch, seq_len, channels)
        :param x_lengths: Int tensor of the lengths of x size (batch,)

        :return: Predicted durations size (batch, seq_len), True=padded mask of x size (batch, seq_len), duration hidden states
        """
        # Generate the appropriate mask for attention
        max_length = x.size(1)
        mask = torch.arange(max_length).expand(len(x_lengths), max_length).to(x_lengths.device)
        mask = mask >= x_lengths.unsqueeze(1)

        # Create an attention mask (batch, 1, seq_len, seq_len)
        attention_mask = mask_to_attention_mask(mask)

        if self.bidirectional:
            x_orig = x.clone()

        # Pass input through the TCNAttention layer
        x = self.tcn_attention(x, attention_mask)

        if self.bidirectional:
            # Reverse input and mask for backward processing
            x_reversed = x_orig.flip(dims=[1])
            mask_reversed = mask.flip(dims=[1])
            reverse_attention_mask = mask_to_attention_mask(mask_reversed)

            x_reversed = self.backwards_tcn_attention(x_reversed, reverse_attention_mask)
            x_reversed = self.backwards_drop(x_reversed)

            # flip back to align with x
            x_reversed = x_reversed.flip(dims=[1])

            # cat and project back to normal
            x = torch.cat((x, x_reversed), dim=-1)
            x = self.fw_projection(x)

        x = self.final_drop(x)

        durations = self.linear_projection(x)
        durations = durations.squeeze(-1)
        durations = durations.masked_fill(mask, 0)

        return durations, mask, x
