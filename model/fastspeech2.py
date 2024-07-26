import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Encoder, Decoder, PostNet
from .modules import VarianceAdaptor,  AlignmentEncoder, sequence_mask, binarize_attention_parallel
from utils.tools import get_mask_from_lengths
from .submodels import TextEncoder, SpectrogramDecoder
from text.symbols import symbols





class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.text_encoder = TextEncoder(
            len(symbols) + 1,
            model_config["transformer"]["encoder_hidden"],
            model_config["transformer"]["encoder_head"],
            model_config["transformer"]["encoder_layer"],
            2,
            model_config["transformer"]["encoder_dropout"],
            1.5,
            2,
        )
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )

        self.postnet = PostNet(n_mel_channels=preprocess_config["preprocessing"]["mel"]["n_mel_channels"])

        self.aligner = AlignmentEncoder(n_mel_channels=preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
                                        n_text_channels=model_config["transformer"]["encoder_hidden"],
                                        n_att_channels=preprocess_config["preprocessing"]["mel"]["n_mel_channels"], temperature=0.0005)

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

    @torch.jit.unused
    def run_aligner(self, text_emb, text_len, text_mask, spect, spect_len, attn_prior):
      text_emb = text_emb.permute(0, 2, 1)
      text_mask = text_mask.permute(0, 2, 1) # [b, 1, mxlen] => [b, mxlen, 1]
      spect = spect.permute(0,2,1)  #[b, mel_len, channels] => [b, channels, mel_len]
      attn_soft, attn_logprob = self.aligner(
                                # note: text_mask is MASK=TRUE, do NOT invert it!!!!
          spect, text_emb, mask=text_mask, attn_prior=attn_prior,conditioning=None
      )
      attn_hard = binarize_attention_parallel(attn_soft, text_len, spect_len)
      attn_hard_dur = attn_hard.sum(2)
      return attn_soft, attn_logprob, attn_hard, attn_hard_dur

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )

        output = self.text_encoder(texts, src_lens)

        # src_masks -> [batch, mxlen] => [batch, 1, mxlen]
        if mels is not None:
            attn_soft, attn_logprob, attn_hard, attn_hard_dur = self.run_aligner(output, src_lens, src_masks.unsqueeze(1), mels,
                                                                             mel_lens, None)
            total_durs = attn_hard_dur.sum(1)
        else:
            attn_soft, attn_logprob, attn_hard, attn_hard_dur, total_durs = torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1), None



        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            src_lens,
            mel_masks,
            max_mel_len,
            p_targets,
            e_targets,
            total_durs,
            p_control,
            e_control,
            d_control,
        )

        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            attn_logprob,
            attn_hard,
            attn_soft,
            total_durs,
        )

    def infer(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        p_control=1.0,
        e_control=1.0,
        d_control=1.0,
    ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = None
        

        output = self.text_encoder(texts, src_lens)

        attn_soft, attn_logprob, attn_hard, attn_hard_dur = torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1)

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )

        (
            output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            src_lens,
            mel_masks,
            None,
            None,
            None,
            None,
            p_control,
            e_control,
            d_control,
        )


        output, mel_masks = self.decoder(output, mel_masks)
        output = self.mel_linear(output)

        postnet_output = self.postnet(output) + output

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            attn_logprob,
            attn_hard,
            attn_soft,
            torch.zeros(1),
        )