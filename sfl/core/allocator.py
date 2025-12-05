"""
Tau Allocator - Resource allocation for the Synthetic Field.

The allocator determines how much compute (τ) each pattern receives
based on its value creation and contribution.
"""

import torch
from typing import Dict, Optional

from sfl.core.types import AllocSignal, UncertaintyBundle, ContributionScore


class TauAllocator:
    """
    Allocates τ (compute time) to patterns based on value and contribution.

    τ (tau) is the proper time of each pattern:
    - τ_rate = 0: frozen (no compute)
    - τ_rate ∈ (0,1): slowed down
    - τ_rate = 1: nominal speed
    - τ_rate > 1: accelerated
    """

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}

        self.base_tau_rate = config.get("base_tau_rate", 1.0)
        self.max_tau_rate = config.get("max_tau_rate", 10.0)
        self.min_tau_rate = config.get("min_tau_rate", 0.1)

        # Weights for different factors
        self.gradient_weight = config.get("gradient_weight", 0.3)
        self.contribution_weight = config.get("contribution_weight", 0.3)
        self.uncertainty_penalty = config.get("uncertainty_penalty", 0.2)
        self.value_weight = config.get("value_weight", 0.2)

        # Compliance settings
        self.default_compliance_window = config.get("compliance_window", 100)
        self.default_penalty = config.get("non_compliance_penalty", 0.1)

    def compute(
        self,
        nh_id: bytes,
        param_grad: torch.Tensor,
        uncertainty: UncertaintyBundle,
        contribution: ContributionScore,
    ) -> AllocSignal:
        """
        Compute allocation signal for a pattern.

        Args:
            nh_id: Pattern identifier
            param_grad: Current gradient tensor
            uncertainty: Uncertainty bundle
            contribution: Contribution score

        Returns:
            AllocSignal with τ rate and other allocation info
        """
        # Base rate
        tau_rate = self.base_tau_rate

        # Gradient magnitude bonus
        grad_magnitude = param_grad.abs().mean().item()
        grad_bonus = min(grad_magnitude * self.gradient_weight, 1.0)
        tau_rate += grad_bonus

        # Contribution bonus
        contrib_bonus = self._contribution_bonus(contribution)
        tau_rate += contrib_bonus * self.contribution_weight

        # Uncertainty penalty
        uncertainty_penalty = uncertainty.scalar * self.uncertainty_penalty
        tau_rate -= uncertainty_penalty

        # Value-based adjustment (would come from critics in full implementation)
        # For now, use gradient direction as proxy
        value_adjustment = self._estimate_value(param_grad)
        tau_rate += value_adjustment * self.value_weight

        # Clamp to valid range
        tau_rate = max(self.min_tau_rate, min(self.max_tau_rate, tau_rate))

        # Determine allowed environments based on contribution
        allowed_envs = self._get_allowed_envs(contribution)

        return AllocSignal(
            tau_rate=tau_rate,
            allowed_envs=allowed_envs,
            compliance_window=self.default_compliance_window,
            non_compliance_penalty=self.default_penalty,
        )

    def _contribution_bonus(self, contribution: ContributionScore) -> float:
        """Calculate bonus based on contribution."""
        # More reports = more bonus (logarithmic)
        report_bonus = min(0.5, contribution.reports_submitted / 1000)

        # Data uniqueness bonus
        uniqueness_bonus = contribution.data_uniqueness * 0.3

        # Access level bonus
        access_bonus = contribution.access_level * 0.1

        return report_bonus + uniqueness_bonus + access_bonus

    def _estimate_value(self, param_grad: torch.Tensor) -> float:
        """
        Estimate value from gradient.

        Positive gradient direction suggests value creation.
        """
        # Use mean of positive gradients as value proxy
        positive_grad = param_grad[param_grad > 0]
        if len(positive_grad) == 0:
            return 0.0
        return min(positive_grad.mean().item(), 1.0)

    def _get_allowed_envs(self, contribution: ContributionScore) -> list:
        """
        Determine allowed environments based on contribution level.

        Higher contribution = more environments accessible.
        """
        # Environment IDs are just examples
        base_envs = [b"env:default"]

        if contribution.access_level >= 1:
            base_envs.append(b"env:compute")
        if contribution.access_level >= 2:
            base_envs.append(b"env:data")
        if contribution.access_level >= 3:
            base_envs.append(b"env:premium")

        return base_envs

    def rebalance(
        self, patterns: Dict[bytes, float], total_tau: float
    ) -> Dict[bytes, float]:
        """
        Rebalance τ allocation across all patterns.

        Ensures total allocation doesn't exceed available compute.

        Args:
            patterns: Dict of pattern_id -> requested_tau_rate
            total_tau: Total available τ

        Returns:
            Dict of pattern_id -> allocated_tau_rate
        """
        if not patterns:
            return {}

        total_requested = sum(patterns.values())

        if total_requested <= total_tau:
            # Enough τ for everyone
            return patterns

        # Scale down proportionally
        scale = total_tau / total_requested
        return {nh_id: rate * scale for nh_id, rate in patterns.items()}

    def freeze_pattern(self, current_signal: AllocSignal) -> AllocSignal:
        """Create a frozen allocation signal."""
        return AllocSignal(
            tau_rate=0.0,
            allowed_envs=[],
            compliance_window=current_signal.compliance_window,
            non_compliance_penalty=current_signal.non_compliance_penalty * 2,
        )

    def boost_pattern(
        self, current_signal: AllocSignal, multiplier: float = 2.0
    ) -> AllocSignal:
        """Boost a pattern's τ rate."""
        return AllocSignal(
            tau_rate=min(current_signal.tau_rate * multiplier, self.max_tau_rate),
            allowed_envs=current_signal.allowed_envs,
            compliance_window=current_signal.compliance_window,
            non_compliance_penalty=current_signal.non_compliance_penalty,
        )
