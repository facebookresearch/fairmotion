# Copyright (c) Facebook, Inc. and its affiliates.

import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(
        self, input_dim=None, hidden_dim=1024, num_layers=1, lstm=None,
    ):
        """LSTMEncoder encodes input vector using LSTM cells.

        Attributes:
            input_dim: Size of input vector
            hidden_dim: Size of hidden state vector
            num_layers: Number of layers of LSTM units
            lstm: Optional; If provided, the lstm cell will be used in the
                encoder. This is useful for sharing lstm parameters with
                decoder.
        """
        super(LSTMEncoder, self).__init__()
        self.lstm = lstm
        if not lstm:
            assert input_dim is not None
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
            )

    def forward(self, input):
        """
        Input:
            input: Input vector to be encoded.
                Expected shape is (batch_size, seq_len, input_dim)
        """
        input = input.transpose(0, 1)
        outputs, (lstm_hidden, lstm_cell) = self.lstm(input)
        return lstm_hidden, lstm_cell, outputs.transpose(0, 1)
