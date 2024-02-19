import torch

__all__ = ['Window', 'Dataset', 'EmbeddingModel']

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else
    ("cuda"
     if torch.cuda.is_available()
     else
     "cpu")
)
