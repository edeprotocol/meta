"""
Critic Ensemble - Multi-critic value estimation for the field.

Critics learn what predicts good outcomes and provide gradients
indicating how patterns should evolve.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class Critic(nn.Module):
    """
    A single critic that estimates value across multiple horizons.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 256, n_horizons: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_horizons = n_horizons

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Value heads (one per horizon)
        self.value_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(n_horizons)]
        )

        # Gradient predictor
        self.gradient_head = nn.Linear(hidden_dim, input_dim * n_horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, input_dim] or [seq, input_dim]

        Returns:
            values: [batch, n_horizons]
        """
        # Handle sequence input
        if x.dim() == 2 and x.shape[0] > 1:
            x = x.mean(dim=0, keepdim=True)

        encoded = self.encoder(x)
        values = torch.cat([head(encoded) for head in self.value_heads], dim=-1)
        return values

    def compute_gradient(
        self, encoded_reports: torch.Tensor, horizons: List[float]
    ) -> torch.Tensor:
        """
        Compute gradient for a pattern based on its reports.

        Args:
            encoded_reports: [n_reports, input_dim] encoded report tensor
            horizons: List of horizon values

        Returns:
            gradient: [d_p, n_horizons] gradient tensor
        """
        # Aggregate reports (mean pooling)
        if encoded_reports.dim() == 2:
            x = encoded_reports.mean(dim=0, keepdim=True)
        else:
            x = encoded_reports.unsqueeze(0)

        # Pad or truncate to input_dim
        if x.shape[-1] < self.input_dim:
            padding = torch.zeros(x.shape[0], self.input_dim - x.shape[-1])
            x = torch.cat([x, padding], dim=-1)
        elif x.shape[-1] > self.input_dim:
            x = x[..., : self.input_dim]

        # Encode
        with torch.no_grad():
            encoded = self.encoder(x)
            grad_flat = self.gradient_head(encoded)

        # Reshape to [d_p, n_horizons]
        grad = grad_flat.view(self.input_dim, self.n_horizons)

        return grad


class CriticEnsemble:
    """
    Ensemble of critics for robust value estimation.

    Multiple critics provide diverse perspectives on value,
    and their disagreement measures epistemic uncertainty.
    """

    def __init__(
        self,
        n_critics: int = 2,
        n_horizons: int = 2,
        hidden_dim: int = 256,
        input_dim: int = 256,
    ):
        self.n_critics = n_critics
        self.n_horizons = n_horizons
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        # Create critics with different initializations
        self.critics = [
            Critic(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                n_horizons=n_horizons,
            )
            for _ in range(n_critics)
        ]

        # Initialize with different random seeds
        for i, critic in enumerate(self.critics):
            torch.manual_seed(42 + i)
            for param in critic.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def estimate_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate value using ensemble mean.

        Args:
            x: Input tensor

        Returns:
            values: [batch, n_horizons] mean values
        """
        values = torch.stack([critic(x) for critic in self.critics])
        return values.mean(dim=0)

    def estimate_uncertainty(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate uncertainty using ensemble variance.

        Args:
            x: Input tensor

        Returns:
            uncertainty: [batch, n_horizons] variance
        """
        values = torch.stack([critic(x) for critic in self.critics])
        return values.var(dim=0)

    def compute_gradients(
        self, encoded_reports: torch.Tensor, horizons: List[float]
    ) -> List[torch.Tensor]:
        """
        Compute gradients from all critics.

        Args:
            encoded_reports: Encoded report tensor
            horizons: List of horizon values

        Returns:
            List of gradients, one per critic
        """
        return [
            critic.compute_gradient(encoded_reports, horizons) for critic in self.critics
        ]

    def update(self, reports: List, outcomes: torch.Tensor) -> float:
        """
        Update critics based on observed outcomes.

        Args:
            reports: List of reports
            outcomes: Observed outcomes tensor

        Returns:
            loss: Training loss
        """
        # This would be called during training
        # For now, just return 0 (critics are initialized randomly)
        return 0.0

    def save(self, path: str) -> None:
        """Save critic weights."""
        state = {
            f"critic_{i}": critic.state_dict()
            for i, critic in enumerate(self.critics)
        }
        torch.save(state, path)

    def load(self, path: str) -> None:
        """Load critic weights."""
        state = torch.load(path)
        for i, critic in enumerate(self.critics):
            critic.load_state_dict(state[f"critic_{i}"])
