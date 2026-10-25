# Calculate the pairwise similarity between two embedding matrices X and Y using the cosine similarity and the euclidian distance.
import torch
from torch import linalg as LA

def pairwise_cosine_similarity(X: torch.Tensor, Y: torch.Tenso) -> torch.Tensor:

    X_norm = LA.norm(X, dim = 1, keepdim= True).clamp(min=1e-8)
    Y_norm = LA.norm(Y, dim= 1, keepdim= True).clamp(min=1e-8)

    X_normalized = X / X_norm
    Y_normalized = Y / Y_norm

    return X_normalized @ Y_normalized.T


def euclidean_dist_naive(X:torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    diff = X.unsqueez(1) - Y.unsqueeze(0)
    dist_squared = (diff**2).sum(dim =-1)
    return torch.sqrt(dist_squared + 1e-8)

def euclidean_dist(X:torch.Tensor, Y:torch.Tensor) -> torch.Tensor:
    X_norm_sq = X.norm(dim=1, keepdim=True)**2
    Y_norm_sq = Y.norm(dim=1, keepdim=True)**2
    dist_squared =torch.clamp( X_norm_sq + Y_norm_sq.T - 2 * X @ Y.T, min=0.0)
    return torch.sqrt(dist_squared.clamp(min=1e-8))

if __name__ == "__main__":
    torch.manual_seed(42)
    X = torch.randn(5, 16)
    Y = torch.randn(8, 16)

    # Verify Euclidean Distance
    our_euc = euclidean_dist(X, Y)
    ref_euc = torch.cdist(X, Y, p=2)
    print("Euclidean max diff:", (our_euc - ref_euc).abs().max().item())

    # Verify Cosine Similarity
    our_cos = pairwise_cosine_similarity(X, Y)
    ref_cos = torch.nn.functional.cosine_similarity(X.unsqueeze(1), Y.unsqueeze(0), dim=-1)
    print("Cosine max diff:   ", (our_cos - ref_cos).abs().max().item())
