"""
Synthetic Field Kernel - The heart of the SFL.

This is the discrete emulator of the computational field.
"""

import hashlib
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Callable
import torch

from sfl.core.types import (
    Report,
    GradientPacket,
    AllocSignal,
    UncertaintyBundle,
    ContributionScore,
    LifecycleEvent,
    LifecycleEventType,
    PatternInfo,
)
from sfl.core.critics import CriticEnsemble
from sfl.core.memory import FieldMemory
from sfl.core.allocator import TauAllocator


class SyntheticFieldKernel:
    """
    The core of the Synthetic Field Layer.

    Maintains field state and computes gradients.

    Usage:
        kernel = SyntheticFieldKernel(config)

        # Register a pattern
        nh_id = kernel.register(param_shape=(128,))

        # Report activity
        kernel.report(Report(...))

        # Pull gradient
        gradient = kernel.pull_gradient(nh_id)
    """

    def __init__(self, config: Optional[dict] = None):
        config = config or {}

        # Horizons for multi-timescale gradients
        self.horizons = config.get("horizons", [1.0, 10.0])
        self.n_horizons = len(self.horizons)

        # Components
        self.critics = CriticEnsemble(
            n_critics=config.get("n_critics", 2),
            n_horizons=self.n_horizons,
            hidden_dim=config.get("critic_hidden_dim", 256),
        )
        self.memory = FieldMemory(config.get("memory_path", "./field_memory"))
        self.allocator = TauAllocator(config.get("allocation", {}))

        # State
        self.patterns: Dict[bytes, PatternInfo] = {}
        self.report_buffer: Dict[bytes, List[Report]] = defaultdict(list)
        self.gradient_cache: Dict[bytes, GradientPacket] = {}
        self.contributions: Dict[bytes, ContributionScore] = {}
        self.lifecycle_events: List[LifecycleEvent] = []

        # Callbacks
        self._on_gradient_update: List[Callable[[bytes, GradientPacket], None]] = []
        self._on_lifecycle_event: List[Callable[[LifecycleEvent], None]] = []

    def register(
        self,
        param_shape: Tuple[int, ...],
        lineage: Optional[List[bytes]] = None,
        metadata: Optional[Dict] = None,
    ) -> bytes:
        """
        Register a new pattern in the field.

        Args:
            param_shape: Shape of the pattern's parameters
            lineage: List of ancestor pattern IDs
            metadata: Optional metadata about the pattern

        Returns:
            nh_id: The pattern's unique identifier
        """
        lineage = lineage or []
        metadata = metadata or {}
        lineage_root = lineage[0] if lineage else b"\x00" * 32
        timestamp = int(time.time() * 1000)
        state_hash = hashlib.sha256(str(param_shape).encode()).digest()

        # Generate nh_id
        nh_id = hashlib.sha256(
            lineage_root + timestamp.to_bytes(8, "big") + state_hash
        ).digest()

        # Initialize pattern info
        self.patterns[nh_id] = PatternInfo(
            nh_id=nh_id,
            param_shape=param_shape,
            lineage=lineage,
            created_at=timestamp,
            tau_rate=1.0,
            status="active",
            metadata=metadata,
        )

        # Initialize contribution
        self.contributions[nh_id] = ContributionScore()

        # Emit lifecycle event
        self._emit_lifecycle_event(
            LifecycleEvent(
                nh_id=nh_id,
                event_type=LifecycleEventType.EMERGE,
                timestamp=timestamp,
                parent_ids=lineage,
                metadata=metadata,
            )
        )

        return nh_id

    def report(self, report: Report) -> None:
        """
        Receive a report from a pattern.

        Args:
            report: The report containing state, action, cost, outcome
        """
        nh_id = report.nh_id

        if nh_id not in self.patterns:
            raise ValueError(f"Unknown pattern: {nh_id.hex()}")

        pattern = self.patterns[nh_id]
        if pattern.status != "active":
            raise ValueError(f"Pattern {nh_id.hex()[:16]} is {pattern.status}")

        # Add to buffer
        self.report_buffer[nh_id].append(report)

        # Integrate into memory
        self.memory.integrate(report)

        # Update contribution
        self._update_contribution(nh_id, report)

        # Recompute gradient if needed
        if self._should_update_gradient(nh_id):
            self._compute_gradient(nh_id)

    def pull_gradient(self, nh_id: bytes) -> GradientPacket:
        """
        Get the current gradient for a pattern.

        Args:
            nh_id: The pattern's identifier

        Returns:
            GradientPacket with gradient, allocation signal, and uncertainty
        """
        if nh_id not in self.patterns:
            raise ValueError(f"Unknown pattern: {nh_id.hex()}")

        # Compute if not cached
        if nh_id not in self.gradient_cache:
            self._compute_gradient(nh_id)

        return self.gradient_cache[nh_id]

    def get_contribution(self, nh_id: bytes) -> ContributionScore:
        """Get a pattern's contribution score."""
        if nh_id not in self.contributions:
            raise ValueError(f"Unknown pattern: {nh_id.hex()}")
        return self.contributions[nh_id]

    def get_pattern(self, nh_id: bytes) -> PatternInfo:
        """Get pattern information."""
        if nh_id not in self.patterns:
            raise ValueError(f"Unknown pattern: {nh_id.hex()}")
        return self.patterns[nh_id]

    def list_patterns(self, status: Optional[str] = None) -> List[PatternInfo]:
        """List all patterns, optionally filtered by status."""
        patterns = list(self.patterns.values())
        if status:
            patterns = [p for p in patterns if p.status == status]
        return patterns

    def fork(self, parent_nh_id: bytes, n_children: int = 2) -> List[bytes]:
        """
        Fork a pattern into multiple children.

        Args:
            parent_nh_id: The parent pattern's ID
            n_children: Number of children to create

        Returns:
            List of child pattern IDs
        """
        if parent_nh_id not in self.patterns:
            raise ValueError(f"Unknown pattern: {parent_nh_id.hex()}")

        parent = self.patterns[parent_nh_id]
        children = []

        for i in range(n_children):
            child_id = self.register(
                param_shape=parent.param_shape,
                lineage=parent.lineage + [parent_nh_id],
                metadata={"forked_from": parent_nh_id.hex(), "fork_index": i},
            )
            children.append(child_id)

            # Share tau_rate
            self.patterns[child_id].tau_rate = parent.tau_rate / n_children

        # Emit fork event
        self._emit_lifecycle_event(
            LifecycleEvent(
                nh_id=parent_nh_id,
                event_type=LifecycleEventType.FORK,
                timestamp=int(time.time() * 1000),
                child_ids=children,
            )
        )

        return children

    def merge(self, pattern_ids: List[bytes]) -> bytes:
        """
        Merge multiple patterns into one.

        Args:
            pattern_ids: List of pattern IDs to merge

        Returns:
            The merged pattern's ID
        """
        if len(pattern_ids) < 2:
            raise ValueError("Need at least 2 patterns to merge")

        for nh_id in pattern_ids:
            if nh_id not in self.patterns:
                raise ValueError(f"Unknown pattern: {nh_id.hex()}")

        # Use the largest param_shape
        shapes = [self.patterns[nh_id].param_shape for nh_id in pattern_ids]
        merged_shape = max(shapes, key=lambda s: sum(s))

        # Combined tau_rate
        total_tau = sum(self.patterns[nh_id].tau_rate for nh_id in pattern_ids)

        # Create merged pattern
        merged_id = self.register(
            param_shape=merged_shape,
            lineage=pattern_ids,
            metadata={"merged_from": [nh_id.hex() for nh_id in pattern_ids]},
        )
        self.patterns[merged_id].tau_rate = total_tau

        # Emit merge event
        self._emit_lifecycle_event(
            LifecycleEvent(
                nh_id=merged_id,
                event_type=LifecycleEventType.MERGE,
                timestamp=int(time.time() * 1000),
                parent_ids=pattern_ids,
            )
        )

        # Dissolve original patterns
        for nh_id in pattern_ids:
            self._dissolve_internal(nh_id, emit_event=False)

        return merged_id

    def freeze(self, nh_id: bytes) -> None:
        """
        Freeze a pattern (tau_rate = 0).

        The pattern is not deleted but stops receiving compute.
        """
        if nh_id not in self.patterns:
            raise ValueError(f"Unknown pattern: {nh_id.hex()}")

        self.patterns[nh_id].tau_rate = 0.0
        self.patterns[nh_id].status = "frozen"

        self._emit_lifecycle_event(
            LifecycleEvent(
                nh_id=nh_id,
                event_type=LifecycleEventType.FREEZE,
                timestamp=int(time.time() * 1000),
            )
        )

    def dissolve(self, nh_id: bytes) -> None:
        """
        Dissolve a pattern permanently.

        The pattern is archived and removed from the active field.
        """
        self._dissolve_internal(nh_id, emit_event=True)

    def _dissolve_internal(self, nh_id: bytes, emit_event: bool = True) -> None:
        """Internal dissolution logic."""
        if nh_id not in self.patterns:
            raise ValueError(f"Unknown pattern: {nh_id.hex()}")

        # Archive in memory
        self.memory.archive_pattern(nh_id)

        # Emit event
        if emit_event:
            self._emit_lifecycle_event(
                LifecycleEvent(
                    nh_id=nh_id,
                    event_type=LifecycleEventType.DISSOLVE,
                    timestamp=int(time.time() * 1000),
                )
            )

        # Update status
        self.patterns[nh_id].status = "dissolved"

        # Clean up
        if nh_id in self.report_buffer:
            del self.report_buffer[nh_id]
        if nh_id in self.gradient_cache:
            del self.gradient_cache[nh_id]

    def _compute_gradient(self, nh_id: bytes) -> None:
        """Compute gradient for a pattern."""
        reports = self.report_buffer[nh_id]
        if not reports:
            # Return zero gradient
            pattern = self.patterns[nh_id]
            param_dim = sum(pattern.param_shape)
            self.gradient_cache[nh_id] = GradientPacket(
                nh_id=nh_id,
                param_grad=torch.zeros(param_dim, self.n_horizons),
                horizons=self.horizons,
                alloc_signal=AllocSignal(tau_rate=pattern.tau_rate),
                uncertainty=UncertaintyBundle(scalar=1.0),
                critic_ids=[],
                timestamp=int(time.time() * 1000),
            )
            return

        # Encode reports
        encoded = self._encode_reports(reports)

        # Get gradients from critics
        critic_gradients = []
        critic_ids = []

        for critic_id, critic in enumerate(self.critics.critics):
            grad = critic.compute_gradient(encoded, self.horizons)
            critic_gradients.append(grad)
            critic_ids.append(critic_id.to_bytes(4, "big"))

        # Aggregate
        param_grad = torch.stack(critic_gradients).mean(dim=0)

        # Compute uncertainty
        uncertainty = self._compute_uncertainty(critic_gradients)

        # Compute allocation
        contribution = self.contributions[nh_id]
        alloc_signal = self.allocator.compute(nh_id, param_grad, uncertainty, contribution)

        # Cache
        gradient = GradientPacket(
            nh_id=nh_id,
            param_grad=param_grad,
            horizons=self.horizons,
            alloc_signal=alloc_signal,
            uncertainty=uncertainty,
            critic_ids=critic_ids,
            timestamp=int(time.time() * 1000),
        )
        self.gradient_cache[nh_id] = gradient

        # Update pattern tau_rate
        self.patterns[nh_id].tau_rate = alloc_signal.tau_rate

        # Notify callbacks
        for callback in self._on_gradient_update:
            callback(nh_id, gradient)

        # Keep only recent reports
        self.report_buffer[nh_id] = reports[-10:]

    def _encode_reports(self, reports: List[Report]) -> torch.Tensor:
        """Encode reports into a tensor."""
        encoded = []
        for r in reports:
            concat = torch.cat(
                [
                    r.state.flatten(),
                    r.action.flatten(),
                    r.cost.flatten(),
                    r.outcome.flatten(),
                ]
            )
            encoded.append(concat)

        return torch.stack(encoded)

    def _compute_uncertainty(self, critic_gradients: List[torch.Tensor]) -> UncertaintyBundle:
        """Compute structured uncertainty."""
        stacked = torch.stack(critic_gradients)

        # Epistemic = disagreement between critics
        epistemic = stacked.var(dim=0).mean().item()

        # Other uncertainties (stubs for v0)
        aleatoric = 0.0
        model = 0.0
        adversarial = 0.0

        # Scalar = simple aggregation
        scalar = epistemic

        return UncertaintyBundle(
            epistemic=epistemic,
            aleatoric=aleatoric,
            model=model,
            adversarial=adversarial,
            scalar=scalar,
        )

    def _update_contribution(self, nh_id: bytes, report: Report) -> None:
        """Update contribution score after a report."""
        contrib = self.contributions[nh_id]
        contrib.reports_submitted += 1

        # Update uniqueness based on outcome entropy
        outcome_entropy = -torch.sum(
            torch.softmax(report.outcome, dim=0)
            * torch.log_softmax(report.outcome, dim=0)
        ).item()
        contrib.data_uniqueness = (
            contrib.data_uniqueness * 0.99 + outcome_entropy * 0.01
        )

        # Update access level based on reports
        if contrib.reports_submitted >= 1000:
            contrib.access_level = 3
        elif contrib.reports_submitted >= 100:
            contrib.access_level = 2
        elif contrib.reports_submitted >= 10:
            contrib.access_level = 1

    def _should_update_gradient(self, nh_id: bytes) -> bool:
        """Decide if gradient should be recomputed."""
        buffer_size = len(self.report_buffer[nh_id])
        # Update every 5 reports or if buffer is getting large
        return buffer_size >= 5 or buffer_size >= 100

    def _emit_lifecycle_event(self, event: LifecycleEvent) -> None:
        """Emit a lifecycle event."""
        self.lifecycle_events.append(event)
        for callback in self._on_lifecycle_event:
            callback(event)

    def on_gradient_update(self, callback: Callable[[bytes, GradientPacket], None]) -> None:
        """Register a callback for gradient updates."""
        self._on_gradient_update.append(callback)

    def on_lifecycle_event(self, callback: Callable[[LifecycleEvent], None]) -> None:
        """Register a callback for lifecycle events."""
        self._on_lifecycle_event.append(callback)

    def get_stats(self) -> Dict:
        """Get field statistics."""
        return {
            "total_patterns": len(self.patterns),
            "active_patterns": len([p for p in self.patterns.values() if p.status == "active"]),
            "frozen_patterns": len([p for p in self.patterns.values() if p.status == "frozen"]),
            "total_reports": sum(len(b) for b in self.report_buffer.values()),
            "total_lifecycle_events": len(self.lifecycle_events),
            "memory_entries": self.memory.size(),
        }
