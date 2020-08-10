# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(
        self, input_dim=None, hidden_dim=1024, num_layers=1, lstm=None,
    ):
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
        input = input.transpose(0, 1)
        outputs, (lstm_hidden, lstm_cell) = self.lstm(input)
        return lstm_hidden, lstm_cell, outputs.transpose(0, 1)
