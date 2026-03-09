import math

import numpy as np
import torch
import torch.nn as nn


class SoRALinear(nn.Module):
    """Low-rank linear with a learnable gate for sparse LoRA (SoRA).

    output = ((dropout(x) @ A.T) * gate) @ B.T * scaling
    """

    def __init__(self, in_features, out_features, r=8, lora_alpha=16, lora_dropout=0.0):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.gate = nn.Parameter(torch.randn(1, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return ((self.lora_dropout(x) @ self.lora_A.T).mul(self.gate) @ self.lora_B.T) * self.scaling


class SoRAWrappedLinear(nn.Module):
    """Wraps an existing linear layer and adds a parallel SoRA branch."""

    def __init__(self, original_linear, r=8, lora_alpha=16, lora_dropout=0.0):
        super().__init__()
        self.original = original_linear
        self.sora = SoRALinear(
            original_linear.in_features,
            original_linear.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        for p in self.original.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.original(x) + self.sora(x)


class SparseAdamW(torch.optim.AdamW):
    """AdamW with soft-thresholding (L0 proximal) applied after each step."""

    def __init__(self, sparse_lambda=0.1, lambda_schedule=None, max_lambda=None, lambda_num=None, **kwargs):
        super().__init__(**kwargs)
        self.sparse_lambda = sparse_lambda
        self.lambda_idx = 0
        self.lambda_schedule = lambda_schedule
        self._build_lambda_list(max_lambda, lambda_num)

    def _build_lambda_list(self, max_lambda, lambda_num):
        if self.lambda_schedule is None:
            self._lambdas = None
            return
        if isinstance(self.lambda_schedule, list):
            self._lambdas = self.lambda_schedule
            return
        if max_lambda is None or lambda_num is None:
            raise ValueError(
                f"max_lambda and lambda_num are required for schedule '{self.lambda_schedule}', "
                f"got max_lambda={max_lambda}, lambda_num={lambda_num}"
            )
        if self.lambda_schedule == "linear":
            self._lambdas = np.linspace(self.sparse_lambda, max_lambda, lambda_num)
        elif self.lambda_schedule == "log_linear":
            self._lambdas = np.log(np.linspace(np.exp(self.sparse_lambda), np.exp(max_lambda), lambda_num))
        elif self.lambda_schedule == "exp_linear":
            self._lambdas = np.exp(np.linspace(np.log(self.sparse_lambda), np.log(max_lambda), lambda_num))
        else:
            raise ValueError(f"Unknown lambda_schedule: {self.lambda_schedule}")

    def step_lambda(self):
        if self._lambdas is None:
            return
        if self.lambda_idx < len(self._lambdas) - 1:
            self.lambda_idx += 1
            self.sparse_lambda = self._lambdas[self.lambda_idx]
            print(f"[SparseAdamW] lambda={self.sparse_lambda}")

    @torch.no_grad()
    def step(self, closure=None):
        loss = super().step(closure)

        for group in self.param_groups:
            for p in group["params"]:
                if self.sparse_lambda > 0:
                    p.data[p.data > self.sparse_lambda] -= self.sparse_lambda
                    p.data[p.data < -self.sparse_lambda] += self.sparse_lambda
                    p.data[abs(p.data) < self.sparse_lambda] = 0.0

        return loss
