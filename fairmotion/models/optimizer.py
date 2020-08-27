# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import torch.optim as optim


class Opt:
    def __init__(self):
        pass

    def step(self):
        pass

    def rate(self):
        pass

    def epoch_step(self, **kwargs):
        pass


class NoamOpt(Opt):
    "Optim wrapper that implements rate."

    def __init__(self, model, model_size=512, factor=2, warmup=4000):
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
        )

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )


class SGDOpt(Opt):
    def __init__(self, model, lr=0.1):
        self.optimizer = optim.SGD(model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "min", factor=0.5, patience=5
        )
        self.model = model

    def step(self):
        self.optimizer.step()
        # TODO: Without clipping/Manually look at gradient values
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

    def epoch_step(self, val_loss):
        self.scheduler.step(val_loss)


class AdamOpt(Opt):
    def __init__(self, model, lr=0.0001):
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def step(self):
        self.optimizer.step()
