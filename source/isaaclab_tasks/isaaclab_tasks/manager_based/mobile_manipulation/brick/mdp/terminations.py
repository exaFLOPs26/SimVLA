import torch

def open(
    threshold: float = 0.3,  # Pi/4
) -> torch.Tensor:
    # TODO: Replace 0.5 with actual drawer angle from the env
    return torch.tensor([0.5 < 0.3], dtype=torch.bool)
