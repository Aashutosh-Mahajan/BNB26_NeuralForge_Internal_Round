"""Optional anomaly architecture; train only on successful training traces."""
import torch
from torch import nn


class StepAutoencoder(nn.Module):
    def __init__(self, input_dim=412, bottleneck=32):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.GELU(), nn.Linear(128, bottleneck))
        self.decoder = nn.Sequential(nn.Linear(bottleneck, 128), nn.GELU(), nn.Linear(128, input_dim))

    def forward(self, values):
        return self.decoder(self.encoder(values))

    @torch.no_grad()
    def anomaly_score(self, values):
        return (values - self(values)).square().mean(-1)
