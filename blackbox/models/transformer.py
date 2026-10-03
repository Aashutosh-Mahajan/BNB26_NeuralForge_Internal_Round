"""M1 Step Blame Transformer: the main diagnosis model (60% of the ensemble weight).

Trained by ``blackbox.models.train``; weights live in ``data/models/transformer.pt``.
Inference on runs longer than ``max_steps`` uses overlapping windows (see ensemble.py).
"""
from __future__ import annotations
import torch
from torch import nn


class StepBlameTransformer(nn.Module):
    def __init__(self, input_dim=412, hidden_dim=128, num_heads=4, layers=4, max_steps=256):
        super().__init__()
        self.project = nn.Linear(input_dim, hidden_dim)
        self.position = nn.Embedding(max_steps, hidden_dim)
        encoder = nn.TransformerEncoderLayer(hidden_dim, num_heads, dim_feedforward=hidden_dim * 4,
                                             dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder, layers)
        self.failure_head = nn.Linear(hidden_dim, 1)
        self.blame_head = nn.Linear(hidden_dim, 1)

    def forward(self, steps, mask):
        """steps [batch,time,features], mask [batch,time] true for valid steps."""
        if steps.shape[1] > self.position.num_embeddings:
            raise ValueError("Trace exceeds the configured maximum step count.")
        if not torch.all(mask.any(dim=1)):
            raise ValueError("Every trace must contain at least one valid step.")
        positions = torch.arange(steps.shape[1], device=steps.device)
        encoded = self.encoder(self.project(steps) + self.position(positions),
                               src_key_padding_mask=~mask.bool())
        pooled = (encoded * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        failure_logits = self.failure_head(pooled).squeeze(-1)
        blame_logits = self.blame_head(encoded).squeeze(-1).masked_fill(~mask.bool(), -torch.inf)
        return {"p_fail": torch.sigmoid(failure_logits), "blame": torch.softmax(blame_logits, dim=1),
                "failure_logits": failure_logits, "blame_logits": blame_logits}


def localization_loss(prediction, failed, guilty_indices, mask, sigma=1.0, blame_weight=1.0):
    """BCE + Gaussian-smoothed localization CE for failed examples only."""
    loss = nn.functional.binary_cross_entropy_with_logits(prediction["failure_logits"], failed.float())
    selected = failed.bool() & (guilty_indices >= 0)
    if selected.any():
        logits = prediction["blame_logits"][selected]
        valid = mask[selected].bool()
        positions = torch.arange(logits.shape[1], device=logits.device)
        target = torch.exp(-0.5 * ((positions - guilty_indices[selected, None]) / sigma) ** 2) * valid
        target /= target.sum(1, keepdim=True).clamp_min(1e-12)
        log_prob = nn.functional.log_softmax(logits, dim=1).masked_fill(~valid, 0)
        loss += blame_weight * -(target * log_prob).sum(1).mean()
    return loss
