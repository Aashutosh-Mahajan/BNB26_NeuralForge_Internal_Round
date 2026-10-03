"""M3: step autoencoder trained only on successful runs; reconstruction error = anomaly."""
import numpy as np
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


def train_autoencoder(x: np.ndarray, epochs=40, seed=42, device="cpu", input_dim=412):
    torch.manual_seed(seed)
    model = StepAutoencoder(input_dim).to(device)
    data = torch.tensor(x, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(data), generator=generator)
        for i in range(0, len(data), 256):
            batch = data[order[i:i + 256].to(device)]
            loss = (model(batch) - batch).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model
