# Copyright (c) Facebook, Inc. and its affiliates.

import numpy as np
import pickle
import torch
import torch.utils.data as data
from fairmotion.utils import constants


class Dataset(data.Dataset):
    def __init__(self, dataset_path, device, mean=None, std=None):
        self.src_seqs, self.tgt_seqs = pickle.load(open(dataset_path, "rb"))
        if mean is None or std is None:
            self.mean = np.mean(self.src_seqs, axis=(0, 1))
            self.std = np.std(self.src_seqs, axis=(0, 1))
        else:
            self.mean = mean
            self.std = std
        self.num_total_seqs = len(self.src_seqs)
        self.device = device

    def __getitem__(self, index):
        """Returns one data pair (source, target)."""
        src_seq = (self.src_seqs[index] - self.mean)/(
            self.std + constants.EPSILON
        )
        tgt_seq = (self.tgt_seqs[index] - self.mean)/(
            self.std + constants.EPSILON
        )
        src_seq = torch.Tensor(src_seq).to(device=self.device).double()
        tgt_seq = torch.Tensor(tgt_seq).to(device=self.device).double()
        return src_seq, tgt_seq

    def __len__(self):
        return self.num_total_seqs


def get_loader(
    dataset_path, batch_size=100, device='cuda', mean=None, std=None,
    shuffle=False,
):
    """Returns data loader for custom dataset.
    Args:
        dataset_path: path to pickled numpy dataset
        device: Device in which data is loaded -- 'cpu' or 'cuda'
        batch_size: mini-batch size.
    Returns:
        data_loader: data loader for custom dataset.
    """
    # build a custom dataset
    dataset = Dataset(dataset_path, device, mean, std)

    # data loader for custom dataset
    # this will return (src_seqs, tgt_seqs) for each iteration
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return data_loader
