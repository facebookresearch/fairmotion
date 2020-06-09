import copy
import torch
import torch.nn as nn
from torch.nn import LayerNorm, MultiheadAttention
from torch.nn.init import xavier_uniform_

from mocap_processing.models import positional_encoding as pe


class STTransformerLayer(nn.Module):
    def __init__(
        self, hidden_dim, dropout=0.1, attention_heads=8, feedforward_size=256,
        num_joints=24, window=120, parallel_attention=4, device="cuda",
    ):
        super(STTransformerLayer, self).__init__()

        self.spatial_attn = MultiheadAttention(
            hidden_dim, attention_heads, dropout=dropout,
        )
        self.spatial_norm = LayerNorm(hidden_dim)
        self.spatial_dropout = nn.Dropout(p=dropout)

        # The parameters of temporal attention MHA are separate for each joint.
        self.parallel_attention = parallel_attention
        self.temporal_attn = _get_clones(
            MultiheadAttention(
                hidden_dim, attention_heads, dropout=dropout
            ),
            self.parallel_attention,
        )
        self.num_devices = torch.cuda.device_count()
        self.num_joints = num_joints

        self.temporal_norm = LayerNorm(hidden_dim)
        self.temporal_dropout = nn.Dropout(p=dropout)

        self.ffn = nn.Linear(hidden_dim, feedforward_size)
        self.relu = nn.ReLU()
        self.ffn_inv = nn.Linear(feedforward_size, hidden_dim)
        self.ffn_norm = LayerNorm(feedforward_size)
        self.ffn_dropout = nn.Dropout(p=dropout)

        self.window = window
        self.device = device

    def prepare(self):
        """
        Call this method after initializing the model and moving it to
        preferred device. Here, we move specific components of the model to
        separate GPUs to enable model parallelism.
        """
        temporal_mask = self._generate_square_subsequent_mask(self.window)
        self.temporal_masks = [temporal_mask for _ in range(self.num_joints)]
        flattened_temporal_mask = self._generate_flattened_mask(
            sz=self.window,
            num_joints=self.num_joints//self.parallel_attention,
        )
        self.flattened_temporal_masks = [
            flattened_temporal_mask for _ in range(self.parallel_attention)
        ]

        if self.device == "cuda":
            print(f"Using {self.num_devices} CUDA devices")
            self.temporal_masks = []
            self.flattened_temporal_masks = []
            for i in range(self.parallel_attention):
                self.temporal_attn[i] = self.temporal_attn[i].to(
                    device=self.get_device_name(i)
                )
                self.temporal_masks.append(
                    temporal_mask.to(device=self.get_device_name(i))
                )
                self.flattened_temporal_masks.append(
                    flattened_temporal_mask.to(device=self.get_device_name(i))
                )

    def get_device_name(self, num_joint):
        device_id = num_joint % self.num_devices
        return f"cuda:{device_id}"

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

    def _generate_flattened_mask(self, sz, num_joints):
        mask = (
            torch.triu(torch.eye(num_joints).repeat(sz, sz)) == 1
        ).transpose(0, 1)
        mask = mask.float().masked_fill(
            mask == 0, float('-inf')
        ).masked_fill(
            mask == 1, float(0.0)
        )
        return mask

    def forward(self, x, t):
        """
        x has shape [self.window, batch_size, num_joints, embedding size]
        0<= t < self.window
        """
        seq_len, batch_size, num_joints, embedding_size = x.shape
        # transpose to (num_joints, batch_size, embedding_size)
        spatial_summary = x[t].permute(1, 0, 2)
        spatial_dropout = self.spatial_dropout(
            self.spatial_attn(
                spatial_summary, spatial_summary, spatial_summary
            )[0]
        )

        spatial_output = self.spatial_norm(
            spatial_summary + spatial_dropout
        )
        # transpose back to (batch_size, num_joints, embedding_size)
        spatial_output = spatial_output.permute(1, 0, 2)

        # transpose to (num_joints, time, batch_size, embedding_size)
        # temporal_summary = x.permute(2, 0, 1, 3)
        # temporal_output = torch.Tensor().to(device=self.device).double()
        # temporal_outputs = []
        # transpose to (num_joints, time, batch_size, embedding_size)
        # import pdb; pdb.set_trace()
        # temporal_summary = x.permute(2, 0, 1, 3).view(
        #     -1, batch_size, embedding_size,
        # )

        # for j in range(num_joints):
        #     temporal_summary = temporal_summary_origs[j].to(
        #         device=self.get_device_name(j)
        #     )
        #     temporal_embeddings = self.temporal_attn[j](
        #         temporal_summary, temporal_summary,
        #         temporal_summary,
        #         attn_mask=self.flattened_temporal_mask,
        #     )[0].unsqueeze(2).to(device=x.device)
        #     temporal_outputs.append(temporal_embeddings)
        # temporal_output = torch.cat(temporal_outputs, axis=2)
        temporal_summary = x.permute(2, 0, 1, 3).reshape(
            self.parallel_attention, -1, batch_size, embedding_size,
        )
        temporal_summary_currs = []
        for j in range(self.parallel_attention):
            temporal_summary_currs.append(
                temporal_summary[j].to(device=self.get_device_name(j))
            )
        temporal_outputs = []
        for j in range(self.parallel_attention):
            temporal_outputs.append(self.temporal_attn[j](
                temporal_summary_currs[j], temporal_summary_currs[j],
                temporal_summary_currs[j],
                attn_mask=self.flattened_temporal_masks[j]
            )[0])
        for j in range(self.parallel_attention):
            temporal_outputs[j] = temporal_outputs[j].to(
                device=x.device
            )

        temporal_output = torch.cat(temporal_outputs).view(
            num_joints, 120, batch_size, embedding_size
        ).permute(1, 2, 0, 3)

        temporal_output = self.temporal_norm(
            self.temporal_dropout(temporal_output) + x
        )
        # transpose back to (time, batch_size, num_joints, embedding_size)
        # temporal_output = temporal_output.permute(1, 2, 0, 3)

        temporal_output[t] = temporal_output[t] + spatial_output

        ffn = self.relu(self.ffn(temporal_output))
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
                hidden_dim, dropout, attention_heads, feedforward_size,
                device=device,
            ),
            num_layers,
        )

        self.project_to_output = nn.Linear(hidden_dim, input_dim//num_joints)

    def init_weights(self):
        """Initiate parameters in the transformer model."""
        for i in range(len(self.layers)):
            self.layers[i].prepare()

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src, t, max_len=None, teacher_forcing_ratio=0.5):
        """
        t does not have bounds, however we use self.window sliding windows in
        the atttention layers
        """
        # convert src to (T, B, E) format
        src = src.transpose(0, 1)
        total_time, batch_size, _ = src.shape

        if self.training:
            """
            During training, the concatenated source and target sequence forms
            the input to the model. At each time step, the model predicts the
            next pose autoregressively.
            """
            inputs = torch.zeros(self.window, batch_size, src.shape[2]).to(
                device=src.device
            ).double()
            # Assign most recent window of src to inputs tensor
            inputs[:min(self.window, total_time)] = src[
                max(0, total_time - self.window):total_time
            ]
            reshaped_inputs = self.dropout(
                self.pos_encoder(self.embedding(inputs))
            ).view(
                self.window, src.shape[1], self.num_joints, -1
            )

            x = reshaped_inputs
            for num, layer in enumerate(self.layers):
                x = layer(x, min(self.window - 1, t))
            x = self.project_to_output(x)
            # Attention layer represents pose with shape
            # (num_joints, angle_dim). We need to flatten it.
            x = x.view(self.window, src.shape[1], -1)
            # window_t is sliding window index of time t
            window_t = min(self.window - 1, t)
            output = x[window_t] + src[t]

            return output.transpose(0, 1)
        else:
            src_len, batch_size, pose_size = src.shape
            if max_len is None:
                # During inference, we are given source sequence and we predict
                # only the target sequence.
                max_len = 24
            inputs = torch.zeros(
                src_len + max_len, src.shape[1], src.shape[2]
            ).to(src.device).double()
            # Set source as input. We fill the rest of the zero tensor with
            # predicted target poses iteratively later
            inputs[:src_len] = src.double()
            outputs = torch.zeros(max_len, src.shape[1], src.shape[2]).to(
                src.device
            ).double()
            for tgt_t in range(max_len):
                combined_t = src_len + tgt_t
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
                    self.window, batch_size, self.num_joints, -1
                )
                x = reshaped_src
                for layer in self.layers:
                    x = layer(x, self.window - 1)
                x = self.project_to_output(x)
                x = x.view(self.window, batch_size, -1)
                window_t = min(self.window - 1, combined_t)
                output = x[window_t] + inputs[combined_t - 1]

                inputs[combined_t] = output
                outputs[tgt_t] = output

        # transpose back to (B, T, E) format
        return outputs.transpose(0, 1)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
