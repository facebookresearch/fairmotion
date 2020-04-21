import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class DecoderStep(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, lstm=None):
        super(DecoderStep, self).__init__()
        self.lstm = (
            nn.LSTM(input_size=input_dim, hidden_size=hidden_dim)
            if not lstm else lstm
        )
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden=None, cell=None, encoder_outputs=None):
        if (hidden is None) and (cell is None):
            output, (hidden, cell) = self.lstm(input)
        else:
            output, (hidden, cell) = self.lstm(input, (hidden, cell))
        output = output.squeeze(0)
        output = self.out(output)
        return output, hidden, cell


class LSTMDecoder(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim, device="cuda", lstm=None
    ):
        super(LSTMDecoder, self).__init__()
        self.input_dim = input_dim
        self.decoder_step = DecoderStep(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            lstm=lstm
        )
        self.device = device

    def forward(
        self,
        tgt,
        hidden=None,
        cell=None,
        encoder_outputs=None,
        max_len=None,
        teacher_forcing_ratio=0.5,
    ):
        tgt = tgt.transpose(0, 1)
        max_len = max_len if max_len is not None else tgt.shape[0]
        batch_size = tgt.shape[1]

        input = tgt[0, :]
        outputs = torch.zeros(max_len, batch_size, self.input_dim,).to(
            self.device
        )
        for t in range(max_len):
            input = input.unsqueeze(0)
            output, hidden, cell = self.decoder_step(input, hidden, cell)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            input = tgt[t] if teacher_force else output

        outputs = outputs.transpose(0, 1)
        return outputs


class DecoderStepWithAttention(nn.Module):
    def __init__(
        self, input_dim, output_dim, hidden_dim, source_length, device="cuda",
    ):
        super(DecoderStepWithAttention, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.source_length = source_length
        self.device = device

        self.attn = nn.Linear(self.hidden_dim + self.input_dim, self.source_length,)
        self.attn_combine = nn.Linear(self.hidden_dim + self.input_dim, self.input_dim,)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, input, hidden, cell, encoder_outputs):
        attn_weights = F.softmax(self.attn(torch.cat((input, hidden), 2)), dim=2,)
        attn_applied = torch.bmm(attn_weights.transpose(0, 1), encoder_outputs)

        output = torch.cat((input, attn_applied.transpose(0, 1)), 2)
        output = self.attn_combine(output)
        output = F.relu(output)

        if (hidden is None) and (cell is None):
            output, (hidden, cell) = self.lstm(output)
        else:
            output, (hidden, cell) = self.lstm(output, (hidden, cell))
        output = output.squeeze(0)
        output = self.out(output)
        return output, hidden, cell


class LSTMDecoderWithAttention(LSTMDecoder):
    def __init__(
        self, input_dim, output_dim, max_source_length, hidden_dim=128, device="cuda"
    ):
        super(LSTMDecoderWithAttention, self).__init__(
            input_dim, output_dim, hidden_dim, device
        )
        self.decoder_step = DecoderStepWithAttention(
            input_dim, output_dim, hidden_dim, max_source_length
        )
        self.device = device
