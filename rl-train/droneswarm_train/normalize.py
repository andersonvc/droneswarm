"""Running observation and value normalizers (Welford's algorithm)."""

import torch


class RunningMeanStd:
    """Ego observation normalizer using Welford's online algorithm."""

    def __init__(self, dim: int, device: torch.device):
        self.mean = torch.zeros(dim, device=device)
        self.var = torch.ones(dim, device=device)
        self.count = 0.0
        self.device = device

    def update(self, batch: torch.Tensor) -> None:
        """Update statistics with a batch of observations [N, dim]."""
        if batch.shape[0] == 0:
            return
        batch_mean = batch.mean(dim=0)
        batch_var = batch.var(dim=0, correction=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / max(total_count, 1.0)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / max(
            total_count, 1.0
        )
        new_var = m2 / max(total_count, 1.0)

        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input [N, dim], clamped to [-10, 10]."""
        return ((x - self.mean) / (self.var + 1e-8).sqrt()).clamp(-10, 10)

    def state_dict(self) -> dict:
        return {
            "mean": self.mean.cpu().tolist(),
            "var": self.var.cpu().tolist(),
            "count": self.count,
        }

    def load_state_dict(self, d: dict) -> None:
        self.mean = torch.tensor(d["mean"], device=self.device)
        self.var = torch.tensor(d["var"], device=self.device)
        self.count = d["count"]


class ValueNormalizer:
    """PopArt-style value target normalization using Welford's algorithm."""

    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 0.0

    def update(self, values: torch.Tensor) -> None:
        """Update running stats with a batch of return values [N]."""
        if values.numel() == 0:
            return
        batch_mean = values.mean().item()
        batch_var = values.var(correction=0).item()
        batch_count = values.numel()

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / max(total_count, 1.0)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / max(
            total_count, 1.0
        )
        self.var = m2 / max(total_count, 1.0)
        self.count = total_count

    def normalize(self, v: float) -> float:
        return (v - self.mean) / max((self.var + 1e-8) ** 0.5, 1e-8)

    def denormalize(self, v: float) -> float:
        return v * (self.var + 1e-8) ** 0.5 + self.mean
