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
            src_seqs, tgt_seqs = src_seqs.to(device), tgt_seqs.to(device)
            outputs = model(src_seqs, tgt_seqs)
            outputs = outputs.to(dtype=torch.double)
            loss = criterion(outputs, tgt_seqs)
            eval_loss += loss.item()
        return eval_loss / (iterations * batch_size)


def generate(model, src_seqs, tgt_seqs, device):
    """
    Generates output sequences for given input sequences by running forward
    pass through the given model
    """
    model.eval()
    with torch.no_grad():
        src_seqs, tgt_seqs = src_seqs.to(device), tgt_seqs.to(device)
        outputs = model(src_seqs, tgt_seqs)
        outputs = outputs.to(dtype=torch.double)
        return outputs.transpose(0, 1).cpu().data.numpy()
