import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, max_len=None, teacher_forcing_ratio=0.5):
        hidden, cell, outputs = self.encoder(src)
        outputs = self.decoder(
            tgt, hidden, cell, outputs, max_len, teacher_forcing_ratio,
        )
        return outputs
