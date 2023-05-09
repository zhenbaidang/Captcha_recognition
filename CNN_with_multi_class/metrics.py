import torch

def acc(pred, y):
    pred = torch.argmax(pred, dim=-1)
    eq = pred == y
    return eq.float().mean(dtype=torch.float32).item()


def multi_acc(pred, y):
    eq = pred == y
    all_eq = torch.all(eq, dim=-1)
    return torch.mean(all_eq.float(), dtype=torch.float32).item()