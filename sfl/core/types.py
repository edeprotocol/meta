"""
SFL Core Types - Data structures for the Synthetic Field Layer.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
import time
import torch


class LifecycleEventType(Enum):
    """Types of lifecycle events for patterns."""
    EMERGE = "emerge"
    FORK = "fork"
    MERGE = "merge"
    FREEZE = "freeze"
    DISSOLVE = "dissolve"


@dataclass
class Report:
    """
    A Report is a sample of a pattern's state at a given moment.

    This is what patterns send to the field to report their activity.
    The field uses this to compute gradients and allocations.
    """
    nh_id: bytes
    state: torch.Tensor
    action: torch.Tensor
    cost: torch.Tensor
    outcome: torch.Tensor
    timestamp: Optional[int] = None
    lineage: List[bytes] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = int(time.time() * 1000)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "nh_id": self.nh_id.hex(),
            "state": self.state.tolist(),
            "action": self.action.tolist(),
            "cost": self.cost.tolist(),
            "outcome": self.outcome.tolist(),
            "timestamp": self.timestamp,
            "lineage": [l.hex() for l in self.lineage],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Report":
        """Create from dictionary."""
        return cls(
            nh_id=bytes.fromhex(data["nh_id"]),
            state=torch.tensor(data["state"]),
            action=torch.tensor(data["action"]),
            cost=torch.tensor(data["cost"]),
            outcome=torch.tensor(data["outcome"]),
            timestamp=data.get("timestamp"),
            lineage=[bytes.fromhex(l) for l in data.get("lineage", [])],
            metadata=data.get("metadata", {}),
        )


@dataclass
class UncertaintyBundle:
    """
    Structured uncertainty on the gradient.

    epistemic: Disagreement between critics (reducible with more data)
    aleatoric: Estimated noise in the data (irreducible)
    model: Structural uncertainty of the model
    adversarial: Suspicion of manipulation
    scalar: Aggregation for simple consumers
    """
    epistemic: float = 0.0
    aleatoric: float = 0.0
    model: float = 0.0
    adversarial: float = 0.0
    scalar: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "epistemic": self.epistemic,
            "aleatoric": self.aleatoric,
            "model": self.model,
            "adversarial": self.adversarial,
            "scalar": self.scalar,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "UncertaintyBundle":
        return cls(**data)


@dataclass
class AllocSignal:
    """
    Resource allocation signal (proper time).

    tau_rate: Proper time / physical time ratio
        0 = frozen (no compute)
        (0,1) = slowed down
        1 = nominal speed
        >1 = accelerated

    allowed_envs: Environments the pattern is allowed to access
    compliance_window: Time to apply the signal (ms)
    non_compliance_penalty: Penalty for ignoring the signal
    """
    tau_rate: float = 1.0
    allowed_envs: List[bytes] = field(default_factory=list)
    compliance_window: int = 100
    non_compliance_penalty: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tau_rate": self.tau_rate,
            "allowed_envs": [e.hex() for e in self.allowed_envs],
            "compliance_window": self.compliance_window,
            "non_compliance_penalty": self.non_compliance_penalty,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AllocSignal":
        return cls(
            tau_rate=data["tau_rate"],
            allowed_envs=[bytes.fromhex(e) for e in data.get("allowed_envs", [])],
            compliance_window=data.get("compliance_window", 100),
            non_compliance_penalty=data.get("non_compliance_penalty", 0.1),
        )


@dataclass
class GradientPacket:
    """
    The field's response to a pattern.

    This is what patterns receive to know how to evolve.
    """
    nh_id: bytes
    param_grad: torch.Tensor  # Shape: [d_p, H] - gradients per horizon
    horizons: List[float]
    alloc_signal: AllocSignal
    uncertainty: UncertaintyBundle
    critic_ids: List[bytes]
    timestamp: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nh_id": self.nh_id.hex(),
            "param_grad": self.param_grad.tolist(),
            "horizons": self.horizons,
            "alloc_signal": self.alloc_signal.to_dict(),
            "uncertainty": self.uncertainty.to_dict(),
            "critic_ids": [c.hex() for c in self.critic_ids],
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GradientPacket":
        return cls(
            nh_id=bytes.fromhex(data["nh_id"]),
            param_grad=torch.tensor(data["param_grad"]),
            horizons=data["horizons"],
            alloc_signal=AllocSignal.from_dict(data["alloc_signal"]),
            uncertainty=UncertaintyBundle.from_dict(data["uncertainty"]),
            critic_ids=[bytes.fromhex(c) for c in data["critic_ids"]],
            timestamp=data["timestamp"],
        )


@dataclass
class ContributionScore:
    """
    A pattern's contribution to the collective memory.
    """
    reports_submitted: int = 0
    gradient_utility: float = 0.0
    data_uniqueness: float = 0.0
    access_level: int = 0  # 0-3

    def to_dict(self) -> Dict[str, Any]:
        return {
            "reports_submitted": self.reports_submitted,
            "gradient_utility": self.gradient_utility,
            "data_uniqueness": self.data_uniqueness,
            "access_level": self.access_level,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ContributionScore":
        return cls(**data)


@dataclass
class LifecycleEvent:
    """
    An event in the lifecycle of a pattern.
    """
    nh_id: bytes
    event_type: LifecycleEventType
    timestamp: int
    parent_ids: List[bytes] = field(default_factory=list)
    child_ids: List[bytes] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nh_id": self.nh_id.hex(),
            "event_type": self.event_type.value,
            "timestamp": self.timestamp,
            "parent_ids": [p.hex() for p in self.parent_ids],
            "child_ids": [c.hex() for c in self.child_ids],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LifecycleEvent":
        return cls(
            nh_id=bytes.fromhex(data["nh_id"]),
            event_type=LifecycleEventType(data["event_type"]),
            timestamp=data["timestamp"],
            parent_ids=[bytes.fromhex(p) for p in data.get("parent_ids", [])],
            child_ids=[bytes.fromhex(c) for c in data.get("child_ids", [])],
            metadata=data.get("metadata", {}),
        )


@dataclass
class PatternInfo:
    """
    Information about a registered pattern.
    """
    nh_id: bytes
    param_shape: tuple
    lineage: List[bytes]
    created_at: int
    tau_rate: float
    status: str  # "active", "frozen", "dissolved"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nh_id": self.nh_id.hex(),
            "param_shape": list(self.param_shape),
            "lineage": [l.hex() for l in self.lineage],
            "created_at": self.created_at,
            "tau_rate": self.tau_rate,
            "status": self.status,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PatternInfo":
        return cls(
            nh_id=bytes.fromhex(data["nh_id"]),
            param_shape=tuple(data["param_shape"]),
            lineage=[bytes.fromhex(l) for l in data.get("lineage", [])],
            created_at=data["created_at"],
            tau_rate=data["tau_rate"],
            status=data["status"],
            metadata=data.get("metadata", {}),
        )
