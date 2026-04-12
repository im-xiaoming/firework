import torch

def log_softmax(x):
    log_probs = x - torch.max(x, dim=-1, keepdim=True)[0]
    log_probs = log_probs - torch.log(
        torch.sum(torch.exp(log_probs), dim=-1, keepdim=True)
    )
    return log_probs