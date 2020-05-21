import copy
import torch
import torch.nn as nn
from torch.nn import LayerNorm, MultiheadAttention
from torch.nn.init import xavier_uniform_

from mocap_processing.models import positional_encoding as pe


class STTransformerLayer(nn.Module):
    def __init__(
        self, hidden_dim, dropout=0.1, attention_heads=8, feedforward_size=256,
        num_joints=24, window=120, device="cuda",
    ):
        super(STTransformerLayer, self).__init__()

        self.spatial_attn = MultiheadAttention(
            hidden_dim, attention_heads, dropout=dropout,
        )
        self.spatial_norm = LayerNorm(hidden_dim)
        self.spatial_dropout = nn.Dropout(p=dropout)

        # The parameters of temporal attention MHA are separate for each joint.
        self.temporal_attn = [
            MultiheadAttention(
                hidden_dim, attention_heads, dropout=dropout
            ).to(device=device).double()
            for _ in range(num_joints)
        ]
        self.temporal_norm = LayerNorm(hidden_dim)
        self.temporal_dropout = nn.Dropout(p=dropout)

        self.ffn = nn.Linear(hidden_dim, feedforward_size)
        self.relu = nn.ReLU()
        self.ffn_inv = nn.Linear(feedforward_size, hidden_dim)
        self.ffn_norm = LayerNorm(feedforward_size)
        self.ffn_dropout = nn.Dropout(p=dropout)

        self.window = window

    def _generate_square_subsequent_mask(self, sz):
        """
        The mask returned for size=4 looks like this
        [
            [0, -inf, -inf, -inf],
            [0, 0, -inf, -inf],
            [0, 0, 0, -inf],
            [0, 0, 0, 0],
        ]

        The mask value is additive and is used in scaled dot product attention.
        It is added to the dot product of query and key, before the tensor is
        fed to the softmax
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(
            mask == 0, float('-inf')
        ).masked_fill(
            mask == 1, float(0.0)
        )
        return mask

    def forward(self, x, t):
        seq_len, batch_size, num_joints, embedding_size = x.shape

        # transpose to (num_joints, batch_size, embedding_size)
        spatial_summary = x[t].permute(1, 0, 2)
        spatial_summary_clone = spatial_summary.clone()
        spatial_attn, _ = self.spatial_attn(
            spatial_summary, spatial_summary, spatial_summary
        )
        spatial_dropout = self.spatial_dropout(spatial_attn)
        spatial_output = self.spatial_norm(
            spatial_summary_clone + spatial_dropout
        )
        # transpose back to (batch_size, num_joints, embedding_size)
        spatial_output = spatial_output.permute(1, 0, 2)

        # transpose to (num_joints, time, batch_size, embedding_size)
        temporal_summary = x[
            max(0, t - self.window): max(t, self.window)
        ].permute(2, 0, 1, 3)
        temporal_mask = self._generate_square_subsequent_mask(self.window).to(
            device=x.device,
        )
        temporal_outputs = []
        for j in range(num_joints):
            temporal_outputs.append(
                self.temporal_norm(
                    temporal_summary[j] + self.temporal_dropout(
                        self.temporal_attn[j](
                            temporal_summary[j], temporal_summary[j],
                            temporal_summary[j], attn_mask=temporal_mask,
                        )[0]
                    )
                ).unsqueeze(0)
            )
        temporal_output = torch.cat(temporal_outputs)
        # transpose back to (time, batch_size, num_joints, embedding_size)
        temporal_output = temporal_output.permute(1, 2, 0, 3)
        window_t = min(self.window - 1, t)
        temporal_output[window_t] = temporal_output[window_t] + spatial_output
        ffn = temporal_output
        ffn = self.relu(self.ffn(ffn))
        ffn = ffn + self.ffn_dropout(ffn)
        output = self.ffn_inv(ffn)

        return output


class STTransformer(nn.Module):
    """
    This is an implementation of the architecture from the paper 'Attention,
    please: A Spatio-temporal Transformer for 3D Human Motion Prediction; Aksan
    et al.' https://arxiv.org/abs/2004.08692

    See Fig 2 for achitecture overview. The basic components of the model are:
    * Joint embedding projection
    * Positional encoding + Dropout
    * Attention layer consisting of 
        * Temporal attention block that updates a joint’s embedding by looking
          at the past instances of the same joint
        * Spatial block that attends over all the joints in the current frame
        * Feed-forward network
    * Embedding projection to joint space
    * Residual connection from input to output
    """
    def __init__(
        self, input_dim, hidden_dim=128, num_layers=4, dropout=0.1,
        attention_heads=8, feedforward_size=256, num_joints=24, window=120,
        device="cpu",
    ):
        super(STTransformer, self).__init__()
        self.num_joints = num_joints
        self.window = window
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = nn.Linear(input_dim, hidden_dim * num_joints)
        self.pos_encoder = pe.PositionalEncoding(
            hidden_dim * num_joints, dropout
        )
        self.layers = _get_clones(
            STTransformerLayer(
                hidden_dim, dropout, attention_heads, feedforward_size
            ),
            num_layers,
        )

        self.project_to_output = nn.Linear(hidden_dim, input_dim//num_joints)

    def init_weights(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, tgt, max_len=None, teacher_forcing_ratio=0.5):
        # convert src, tgt to (T, B, E) format
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        if self.training:
            """
            During training, the concatenated source and target sequence forms
            the input to the model. At each time step, the model predicts the
            next pose autoregressively.
            """
            concat_inputs = torch.cat((src, tgt))
            # Target sequence is src[1:] + tgt[:]
            max_len = src.shape[0] + tgt.shape[0] - 1
            reshaped_inputs = self.dropout(
                self.pos_encoder(self.embedding(concat_inputs))
            ).view(
                max_len + 1, tgt.shape[1], self.num_joints, -1
            )
            # We store autoregressive predictions by the model in 
            # zero-initialized `outputs` tensor
            outputs = torch.zeros(max_len, src.shape[1], src.shape[2]).to(
                src.device
            )
            for t in range(max_len):
                x = reshaped_inputs
                for layer in self.layers:
                    x = layer(x, t)
                x = self.project_to_output(x)
                # Attention layer represents pose with shape
                # (num_joints, angle_dim). We need to flatten it.
                x = x.view(self.window, src.shape[1], -1)
                # window_t is sliding window index of time t
                window_t = min(self.window - 1, t)
                output = x[window_t] + concat_inputs[t]
                outputs[t] = output
        else:
            if max_len is None:
                # During inference, we are given source sequence and we predict
                # only the target sequence.
                max_len = tgt.shape[0]
            inputs = torch.zeros(
                src.shape[0] + max_len - 1, src.shape[1], src.shape[2]
            ).to(src.device)
            # Set source as input. We fill the rest of the zero tensor with
            # predicted target poses iteratively later
            inputs[:src.shape[0]] = src
            outputs = torch.zeros(max_len, src.shape[1], src.shape[2]).to(
                src.device
            )
            for tgt_t in range(max_len):
                combined_t = src.shape[0] + tgt_t
                # Bounds of sliding window of inputs that are passed to the
                # attention layers
                window_l, window_r = (
                    max(0, combined_t - self.window),
                    max(combined_t, self.window),
                )
                reshaped_src = self.dropout(
                    self.pos_encoder(self.embedding(
                        inputs[window_l: window_r]
                    ))
                ).view(
                    max_len + 1, tgt.shape[1], self.num_joints, -1
                )
                x = reshaped_src
                for layer in self.layers:
                    x = layer(x, t)
                x = self.project_to_output(x)
                x = x.view(self.window, src.shape[1], -1)
                window_t = min(self.window - 1, combined_t)
                output = x[window_t] + inputs[combined_t - 1]

                inputs[combined_t] = output
                outputs[tgt_t] = output

        # transpose back to (B, T, E) format
        return outputs.transpose(0, 1)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
