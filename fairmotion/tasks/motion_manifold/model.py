# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import torch.nn as nn


class MLP(nn.Module):
    """MLP model for learning natural motion. Model returns a naturalness
    score for predicted pose that appears after a sequence of observed poses.

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
        num_observed,
        hidden_dim=256,
        num_layers=1,
        dropout=0.1,
        device="cuda",
    ):
        super(MLP, self).__init__()
        self.num_observed = num_observed
        self.mlp_input = nn.Linear(input_dim, hidden_dim)
        self.mlp_concat = nn.Linear((num_observed + 1)*hidden_dim, hidden_dim)
        self.mlp = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(p=dropout)
        self.project_to_score = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def init_weights(self):
        for name, param in self.named_parameters():
            nn.init.uniform_(param.data, -0.1, 0.1)

    def forward(self, observed, predicted):
        """
        Inputs:
            observed: Tensor of shape (batch_size, num_observed, input_dim)
            predicted: Tensor of shape (batch_size, input_dim)
        """
        observed = self.mlp_input(observed)
        observed = torch.reshape(
            observed, (-1, observed.shape[-2] * observed.shape[-1])
        )
        predicted = self.mlp_input(predicted)
        x = torch.cat((observed, predicted), axis=1)
        x = self.mlp_concat(x)
        for layer in self.mlp:
            x = self.dropout(x)
            x = layer(x)
            x = self.relu(x)

        x = self.project_to_score(x)
        x = torch.sigmoid(x)
        return x
