import pickle
import torch
import torch.utils.data as data


class Dataset(data.Dataset):
    def __init__(self, dataset_path):
        self.src_seqs, self.tgt_seqs = pickle.load(dataset_path, "rb")
        self.num_total_seqs = len(self.src_seqs)

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        src_seq = self.src_seqs[index]
        tgt_seq = self.tgt_seqs[index]
        src_seq = torch.Tensor(src_seq)
        tgt_seq = torch.Tensor(tgt_seq)
        return src_seq, tgt_seq

    def __len__(self):
        return self.num_total_seqs


def get_loader(dataset_path, batch_size=100):
    """Returns data loader for custom dataset.
    Args:
        dataset_path: path to pickled numpy dataset
        batch_size: mini-batch size.
    Returns:
        data_loader: data loader for custom dataset.
    """
    # build a custom dataset
    dataset = Dataset(dataset_path)

    # data loader for custom dataset
    # this will return (src_seqs, tgt_seqs) for each iteration
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    return data_loader
