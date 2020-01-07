import numpy as np
import torch


"""
Evaluate the performance of the model on the provided dataset. Returns average loss over the dataset. 
"""
def eval(model, criterion, sequences):
    model.eval()
    with torch.no_grad():
        batched_test_data_t = torch.from_numpy(np.array(sequences)).to(device='cuda')
        batched_test_target_data_t = batched_test_data_t.transpose(0, 1)
        outputs = model(batched_test_data_t, batched_test_target_data_t)
        outputs = outputs.to(dtype=torch.double)
        loss = criterion(outputs, batched_test_data_t)
        return loss.item()

"""
Generates output sequences for given input sequences by running forward pass through the given model
"""
def generate(model, sequences):
    model.eval()
    with torch.no_grad():
        batched_test_data_t = torch.from_numpy(np.array(sequences)).to(device='cuda')
        batched_test_target_data_t = batched_test_data_t.transpose(0, 1)
        outputs = model(batched_test_data_t, batched_test_target_data_t)
        return outputs.transpose(0, 1).cpu().data.numpy(), batched_test_data_t.cpu().data.numpy()