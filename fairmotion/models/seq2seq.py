# Copyright (c) Facebook, Inc. and its affiliates.

import torch.nn as nn
from fairmotion.models import decoders, encoders


class Seq2Seq(nn.Module):
    """Seq2Seq model for sequence generation. The interface takes predefined
    encoder and decoder as input.

    Attributes:
        encoder: Pre-built encoder
        decoder: Pre-built decoder
    """

    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def forward(self, src, tgt, max_len=None, teacher_forcing_ratio=0.5):
        """
        Inputs:
            src: Source sequence provided as input to the encoder.
                Expected shape: (batch_size, seq_len, input_dim)
            tgt: Target sequence provided as input to the decoder. During
                training, provide reference target sequence. For inference,
                provide only last frame of source.
                Expected shape: (batch_size, seq_len, input_dim)
            max_len: Optional; Length of sequence to be generated. By default,
                the decoder generates sequence with same length as `tgt`
                (training).
            teacher_forcing_ratio: Probability of feeding gold target pose as
                decoder input instead of predicted pose from previous time step
        """
        hidden, cell, outputs = self.encoder(src)
        outputs = self.decoder(
            tgt, hidden, cell, max_len, teacher_forcing_ratio,
        )
        return outputs


class TiedSeq2Seq(nn.Module):
    """Seq2Seq model that uses the same LSTM unit in the encoder and decoder.
    The shared LSTM is created inside this class.

    Attributes:
        input_dim: Size of input vector
        hidden_dim: Size of hidden state vector
        num_layers: Number of layers in the shared LSTM unit
        device: Optional; Device to be used "cuda" or "cpu"
    """

    def __init__(self, input_dim, hidden_dim, num_layers, device):
        super(TiedSeq2Seq, self).__init__()
        tied_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
        ).to(device)
        self.encoder = encoders.LSTMEncoder(
            input_dim=input_dim, lstm=tied_lstm
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
