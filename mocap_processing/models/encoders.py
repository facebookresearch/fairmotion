import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1):
        super(LSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, input):
        outputs, (lstm_hidden, lstm_cell) = self.lstm(input)
        return lstm_hidden, lstm_cell, outputs
