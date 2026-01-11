# model.py
import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Simple fully-connected DQN network.

    Expects input as a flat vector of size `input_dim`.
    Outputs a vector of Q-values of size `output_dim` (one per action).
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims=(128, 128)):
        super().__init__()

        layers = []
        last_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h

        layers.append(nn.Linear(last_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        # Xavier init for linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, input_dim) or anything that can be flattened to that.
        """
        # Just in case input isn't already flat
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)