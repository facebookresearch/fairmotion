import numpy as np
import pickle
import torch
import torch.utils.data as data
from fairmotion.utils import constants


class Dataset(data.Dataset):
    def __init__(self, dataset_path, device):
        self.samples, self.labels = pickle.load(open(dataset_path, "rb"))
        self.mean = np.mean(self.samples, axis=(0, 1))
        self.std = np.std(self.samples, axis=(0, 1))
        self.num_total_seqs = len(self.samples)
        self.device = device

    def __getitem__(self, index):
        """Returns tuple (observed, predicted, score)."""

        observed = torch.Tensor(
            (self.samples[index][:-1] - self.mean) / (
                self.std + constants.EPSILON
            )
        ).to(device=self.device).double()
        predicted = torch.Tensor(
            (self.samples[index][-1] - self.mean) / (
                self.std + constants.EPSILON
            )
        ).to(device=self.device).double()
        label = torch.Tensor(
            [self.labels[index]]
        ).to(device=self.device).double()
        return observed, predicted, label

    def __len__(self):
        return self.num_total_seqs


def get_loader(
    dataset_path,
    batch_size=128,
    device="cuda",
    mean=None,
    std=None,
    shuffle=True,
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
    dataset = Dataset(dataset_path, device)

    # data loader for custom dataset
    # this will return (src_seqs, tgt_seqs) for each iteration
    data_loader = data.DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle,
    )
    return data_loader
