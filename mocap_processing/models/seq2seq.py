import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        hidden, cell, outputs = self.encoder(src)
        outputs = self.decoder(trg, hidden, cell, outputs)
        return outputs
