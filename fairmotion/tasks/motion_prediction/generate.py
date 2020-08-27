# Copyright (c) Facebook, Inc. and its affiliates.

import torch


def eval(model, criterion, dataset, batch_size, device):
    """
    Evaluate the performance of the model on the provided dataset.
    Returns average loss over the dataset.
    """
    model.eval()
    eval_loss = 0
    with torch.no_grad():
        for iterations, (src_seqs, tgt_seqs) in enumerate(dataset):
            max_len = tgt_seqs.shape[1]
            seed_tgt_seqs = src_seqs[:, -1].unsqueeze(1)
            src_seqs, tgt_seqs, seed_tgt_seqs = (
                src_seqs.to(device),
                tgt_seqs.to(device),
                seed_tgt_seqs.to(device),
            )
            outputs = model(
                src_seqs,
                seed_tgt_seqs,
                max_len=max_len,
                teacher_forcing_ratio=0,
            )
            outputs = outputs.double()
            loss = criterion(outputs, tgt_seqs)
            eval_loss += loss.item()
        return eval_loss / ((iterations + 1) * batch_size)


def generate(model, src_seqs, max_len, device):
    """
    Generates output sequences for given input sequences by running forward
    pass through the given model
    """
    model.eval()
    with torch.no_grad():
        tgt_seqs = src_seqs[:, -1].unsqueeze(1)
        src_seqs, tgt_seqs = src_seqs.to(device), tgt_seqs.to(device)
        outputs = model(
            src_seqs, tgt_seqs, max_len=max_len, teacher_forcing_ratio=0
        )
        return outputs.double()
