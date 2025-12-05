"""
Synthetic Field Layer - The Economic Operating System for AGI/ASI

Usage:
    from sfl import SFLClient, Report, GradientPacket

    client = SFLClient("http://localhost:8420")
    pattern_id = client.register(param_shape=(256,))

    client.report(Report(
        nh_id=pattern_id,
        state=torch.randn(32),
        action=torch.randn(8),
        cost=torch.tensor([0.1]),
        outcome=torch.randn(16)
    ))

    gradient = client.pull_gradient(pattern_id)
"""

__version__ = "0.1.0"
__author__ = "EDE Protocol"

from sfl.core.types import (
    Report,
    GradientPacket,
    AllocSignal,
    UncertaintyBundle,
    ContributionScore,
    LifecycleEvent,
    LifecycleEventType,
)
from sfl.core.kernel import SyntheticFieldKernel
from sfl.core.critics import CriticEnsemble
from sfl.core.memory import FieldMemory
from sfl.core.allocator import TauAllocator
from sfl.core.quantum import QuantumCorrelator, EntanglementBond, BellState
from sfl.core.emergence import EmergenceDetector, AlertLevel, AlertType
from sfl.client.sync_client import SFLClient
from sfl.client.async_client import AsyncSFLClient

__all__ = [
    # Types
    "Report",
    "GradientPacket",
    "AllocSignal",
    "UncertaintyBundle",
    "ContributionScore",
    "LifecycleEvent",
    "LifecycleEventType",
    # Core
    "SyntheticFieldKernel",
    "CriticEnsemble",
    "FieldMemory",
    "TauAllocator",
    # Quantum
    "QuantumCorrelator",
    "EntanglementBond",
    "BellState",
    # Emergence
    "EmergenceDetector",
    "AlertLevel",
    "AlertType",
    # Clients
    "SFLClient",
    "AsyncSFLClient",
]
