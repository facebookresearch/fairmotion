# Copyright (c) Facebook, Inc. and its affiliates.

import random
import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(
        self, input_dim, hidden_dim=1024, num_layers=1, dropout=0.1,
        device="cpu",
    ):
        super(RNN, self).__init__()
        # self.project_to_hidden = nn.Linear(input_dim, hidden_dim)
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

    def forward(self, src, tgt, max_len=None, teacher_forcing_ratio=0.5):
        # convert src, tgt to (T, B, E) format
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        # src = self.project_to_hidden(src)
        lstm_input = self.dropout(src)
        state = None
        max_len = tgt.shape[0] if max_len is None else max_len
        # outputs = torch.zeros(
        #     src.shape[0] - 1 + max_len, src.shape[1], src.shape[2]
        # ).to(
        #     src.device
        # )
        for t, input in enumerate(lstm_input[:-1]):
            # teacher_force = random.random() < teacher_forcing_ratio
            # if t > 0 and not teacher_force:
            #     input = output.squeeze(0)
            input = input.unsqueeze(0)
            output, state = self.lstm(input, state)
            output = self.project_to_output(output)
            output = output.squeeze(0)
            # outputs[t] = output

        if self.training:
            # del outputs
            outputs = torch.zeros(max_len, src.shape[1], src.shape[2]).to(
                src.device
            )
            tgt = self.dropout(tgt)
            input = lstm_input[-1]
            for t_gen, input in enumerate(
                torch.cat((lstm_input[-1].unsqueeze(0), tgt[:-1]))
            ):
                teacher_force = random.random() < teacher_forcing_ratio
                if t_gen > 0 and not teacher_force:
                    input = output
                input = input.unsqueeze(0)
                output, state = self.lstm(input, state)
                output = self.project_to_output(output)
                output = output.squeeze(0)
                # outputs[t + 1 + t_gen] = output
                outputs[t_gen] = output
        else:
            # del outputs
            outputs = torch.zeros(max_len, src.shape[1], src.shape[2]).to(
                src.device
            )
            input = lstm_input[-1]
            for t_gen in range(max_len):
                input = input.unsqueeze(0)
                output, state = self.lstm(input, state)
                output = self.project_to_output(output)
                output = output.squeeze(0)
                outputs[t_gen] = output
                input = output

        return outputs.transpose(0, 1)
