# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.nn.init import xavier_uniform_

from fairmotion.models import decoders, encoders


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def forward(self, src, tgt, max_len=None, teacher_forcing_ratio=0.5):
        hidden, cell, outputs = self.encoder(src)
        outputs = self.decoder(
            tgt, hidden, cell, outputs, max_len, teacher_forcing_ratio,
        )
        return outputs


class TiedSeq2Seq(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, device):
        super(TiedSeq2Seq, self).__init__()
        tied_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
        ).to(device)
        self.encoder = encoders.LSTMEncoder(
            input_dim=input_dim,
            lstm=tied_lstm
        ).to(device)
        self.decoder = decoders.LSTMDecoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            device=device,
            lstm=tied_lstm,
        ).to(device)

    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def forward(self, src, tgt, max_len=None, teacher_forcing_ratio=0.5):
        hidden, cell, outputs = self.encoder(src)
        outputs = self.decoder(
            tgt, hidden, cell, outputs, max_len, teacher_forcing_ratio,
        )
        return outputs


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.5, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        # from torch.nn import TransformerDecoder, TransformerDecoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # Use Linear instead of Embedding for continuous valued input
        self.encoder = nn.Linear(ntoken, ninp)
        self.ninp = ninp
        self.decoder = decoders.LSTMDecoder(
            input_dim=ntoken,
            hidden_dim=nhid,
            output_dim=ntoken,
        )
        # self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        for name, param in self.decoder.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def forward(self, src, tgt, max_len=None, teacher_forcing_ratio=None):
        # if self.src_mask is None or self.src_mask.size(0) != len(src):
        #     device = src.device
        #     mask = self._generate_square_subsequent_mask(len(src)).to(device)
        #     self.src_mask = mask

        src = self.encoder(src) * np.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(
            tgt,
            encoder_outputs=output,
            max_len=max_len,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )
        return output


class FullTransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(FullTransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        from torch.nn import TransformerDecoder, TransformerDecoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None

        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layer = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=nlayers,
            norm=LayerNorm(ninp),
        )
        decoder_layer = TransformerDecoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=nlayers,
            norm=LayerNorm(ninp),
        )

        # Use Linear instead of Embedding for continuous valued input
        self.encoder = nn.Linear(ntoken, ninp)
        self.project = nn.Linear(ninp, ntoken)
        self.ninp = ninp

        self._init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(
            mask == 0, float('-inf')
        ).masked_fill(
            mask == 1, float(0.0)
        )
        return mask

    def init_weights(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, tgt, max_len=None, teacher_forcing_ratio=None):
        # Transformer expects src and tgt in format (len, batch_size, dim)
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        # src and tgt are now (T, B, E)
        if max_len is None:
            max_len = tgt.shape[0]

        projected_src = self.encoder(src) * np.sqrt(self.ninp)
        pos_encoded_src = self.pos_encoder(projected_src)
        encoder_output = self.transformer_encoder(pos_encoded_src)
        tgt_mask = self._generate_square_subsequent_mask(tgt.shape[0]).to(
            device=tgt.device,
        )
        if self.training:
            # Use last source pose as first input to decoder
            tgt = torch.cat((src[-1].unsqueeze(0), tgt[:-1]))
            pos_encoder_tgt = self.pos_encoder(
                self.encoder(tgt) * np.sqrt(self.ninp)
            )
            output = self.transformer_decoder(
                pos_encoder_tgt, encoder_output, tgt_mask=tgt_mask,
            )
            output = self.project(output)
        else:
            # greedy decoding
            decoder_input = torch.zeros(
                max_len, src.shape[1], src.shape[-1],
            ).type_as(src.data)
            next_pose = tgt[0].clone()
            for i in range(max_len):
                decoder_input[i] = next_pose
                pos_encoded_input = self.pos_encoder(
                    self.encoder(decoder_input) * np.sqrt(self.ninp)
                )
                decoder_outputs = self.transformer_decoder(
                    pos_encoded_input, encoder_output, tgt_mask=tgt_mask,
                )
                output = self.project(decoder_outputs)
                next_pose = output[i].clone()
                del output
            output = decoder_input
        return output.transpose(0, 1)
