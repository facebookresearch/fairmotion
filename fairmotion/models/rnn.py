# Copyright (c) Facebook, Inc. and its affiliates.

import random
import torch
import torch.nn as nn


class RNN(nn.Module):
    """RNN model for sequence prediction. The model uses a single RNN module to
    take an input pose, and generates a pose prediction for the next time step.

    Attributes:
        input_dim: Size of input vector for each time step
        hidden_dim: RNN hidden size
        num_layers: Number of layers of RNN cells
        dropout: Probability of an element to be zeroed
        device: Device on which to run the RNN module
    """

    def __init__(
        self,
        input_dim,
        hidden_dim=1024,
        num_layers=1,
        dropout=0.1,
        device="cpu",
    ):
        super(RNN, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
        )
        self.project_to_output = nn.Linear(hidden_dim, input_dim)

    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.08, 0.08)

    def run_lstm(
        self,
        inputs,
        outputs,
        max_len=None,
        state=None,
        teacher_forcing_ratio=0,
    ):
        output = torch.zeros(inputs[0].shape).unsqueeze(0)
        if max_len is None:
            max_len = inputs.shape[0]
        else:
            teacher_forcing_ratio = 0

        for t in range(max_len):
            if t >= inputs.shape[0]:
                input = output.unsqueeze(0)
            else:
                input = inputs[t].unsqueeze(0)
            teacher_force = random.random() < teacher_forcing_ratio
            if t > 0 and not teacher_force:
                input = output.unsqueeze(0)
            output, state = self.lstm(input, state)
            output = self.project_to_output(output)
            output = output.squeeze(0)
            outputs[t] = output
        return outputs, state

    def forward(self, src, tgt, max_len=None, teacher_forcing_ratio=0.5):
        """
        Inputs:
            src, tgt: Tensors of shape (batch_size, seq_len, input_dim)
            max_len: Maximum length of sequence to be generated during
                inference. Set None during training.
            teacher_forcing_ratio: Probability of feeding gold target pose as
                decoder input instead of predicted pose from previous time step
        """
        # convert src, tgt to (seq_len, batch_size, input_dim) format
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        lstm_input = self.dropout(src)
        state = None
        # Generate as many poses as in tgt during training
        max_len = tgt.shape[0] if max_len is None else max_len
        encoder_outputs = torch.zeros(src.shape).to(src.device)
        _, state = self.run_lstm(
            lstm_input,
            encoder_outputs,
            state=None,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

        if self.training:
            decoder_outputs = torch.zeros(
                max_len - 1, src.shape[1], src.shape[2]
            ).to(src.device)
            tgt = self.dropout(tgt)
            decoder_outputs, _ = self.run_lstm(
                tgt[:-1],
                decoder_outputs,
                state=state,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )
            outputs = torch.cat((encoder_outputs, decoder_outputs))
        else:
            del encoder_outputs
            outputs = torch.zeros(max_len, src.shape[1], src.shape[2]).to(
                src.device
            )
            inputs = lstm_input[-1].unsqueeze(0)
            outputs, _ = self.run_lstm(
                inputs,
                outputs,
                state=state,
                max_len=max_len,
                teacher_forcing_ratio=0,
            )
        return outputs.transpose(0, 1)
