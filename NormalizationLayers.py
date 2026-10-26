import torch.nn as nn
import torch

class CustomLayerNorm(nn.Module):

    def __init__(self, D: int):

        super().__init__()
        self.eps = 1e-8
        self.weight = nn.Parameter(torch.ones(D,), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(D,), requires_grad=True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        X_f32 = X.float()
        mu = torch.mean(X_f32, dim=-1, keepdim= True)
        var = torch.var(X_f32, dim=-1, keepdim = True, unbiased = False)

        normalized_X =((X_f32 - mu ) *  torch.rsqrt(var + self.eps)) * self.weight + self.bias

        return normalized_X.type_as(X)

class CustomRMSNorm(nn.Module):

    def __init__(self, D: int):

        super().__init__()
        self.eps = 1e-8
        self.weight = nn.Parameter(torch.ones(D,), requires_grad=True)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X_f32 = X.float()
        mu_sq = X_f32.pow(2).mean(dim=-1, keepdim=True)

        RMSnormalized_X = self.weight * (X_f32 * torch.rsqrt(mu_sq + self.eps))

        return RMSnormalized_X.type_as(X)


x = torch.randn(2, 4, 16)
ln_custom = CustomLayerNorm(16)
ln_official = nn.LayerNorm(16)

RMS_custom = CustomRMSNorm(16)
RMS_official = nn.RMSNorm(16)

print(torch.allclose(ln_custom(x), ln_official(x), atol=1e-5))
print(torch.allclose(RMS_custom(x), RMS_official(x), atol=1e-5))
    