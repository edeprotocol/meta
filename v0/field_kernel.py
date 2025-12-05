"""
Synthetic Field Layer - V0 Field Kernel

Émulateur discret du champ computationnel.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import time
import hashlib

from critic_ensemble import CriticEnsemble
from field_memory import FieldMemory
from observation_policy import ObservationPolicy
from lifecycle_manager import LifecycleManager
from contribution_tracker import ContributionTracker
from tau_allocator import TauAllocator


@dataclass
class Report:
    nh_id: bytes
    state: torch.Tensor
    action: torch.Tensor
    cost: torch.Tensor
    outcome: torch.Tensor
    timestamp: int
    lineage: List[bytes] = field(default_factory=list)


@dataclass
class UncertaintyBundle:
    epistemic: float = 0.0
    aleatoric: float = 0.0
    model: float = 0.0
    adversarial: float = 0.0
    scalar: float = 0.0


@dataclass
class AllocSignal:
    tau_rate: float = 1.0
    allowed_envs: List[bytes] = field(default_factory=list)
    compliance_window: int = 100
    non_compliance_penalty: float = 0.1


@dataclass
class GradientPacket:
    nh_id: bytes
    param_grad: torch.Tensor  # Shape: [d_p, H]
    horizons: List[float]
    alloc_signal: AllocSignal
    uncertainty: UncertaintyBundle
    critic_ids: List[bytes]
    timestamp: int


class SyntheticFieldKernel:
    """
    Cœur du Synthetic Field Layer.
    
    Maintient l'état du champ et calcule les gradients.
    """
    
    def __init__(self, config: dict):
        self.config = config
        
        # Horizons par défaut
        self.horizons = config.get('horizons', [1.0, 10.0])
        self.n_horizons = len(self.horizons)
        
        # Composants
        self.critics = CriticEnsemble(
            n_critics=config.get('n_critics', 2),
            n_horizons=self.n_horizons,
            hidden_dim=config.get('critic_hidden_dim', 256)
        )
        self.memory = FieldMemory(config.get('memory_path', './field_memory'))
        self.observation_policy = ObservationPolicy()
        self.lifecycle = LifecycleManager()
        self.contribution = ContributionTracker()
        self.tau_allocator = TauAllocator()
        
        # État
        self.patterns: Dict[bytes, dict] = {}
        self.report_buffer: Dict[bytes, List[Report]] = defaultdict(list)
        self.gradient_cache: Dict[bytes, GradientPacket] = {}
        
    def register(self, 
                 param_shape: Tuple[int, ...],
                 lineage: List[bytes] = None) -> bytes:
        """
        Enregistre un nouveau pattern dans le champ.
        """
        # Générer nh_id
        lineage = lineage or []
        lineage_root = lineage[0] if lineage else b'\x00' * 32
        timestamp = int(time.time() * 1000)
        state_hash = hashlib.sha256(str(param_shape).encode()).digest()
        
        nh_id = hashlib.sha256(
            lineage_root + timestamp.to_bytes(8, 'big') + state_hash
        ).digest()
        
        # Initialiser le pattern
        self.patterns[nh_id] = {
            'param_shape': param_shape,
            'lineage': lineage,
            'created_at': timestamp,
            'tau_rate': 1.0,
            'status': 'active'
        }
        
        # Enregistrer dans lifecycle
        self.lifecycle.emit_emerge(nh_id, lineage)
        
        # Initialiser contribution
        self.contribution.init_pattern(nh_id)
        
        return nh_id
    
    def report(self, report: Report) -> None:
        """
        Reçoit un report d'un pattern.
        """
        nh_id = report.nh_id
        
        if nh_id not in self.patterns:
            raise ValueError(f"Unknown pattern: {nh_id.hex()}")
        
        # Ajouter au buffer
        self.report_buffer[nh_id].append(report)
        
        # Intégrer dans la mémoire
        self.memory.integrate(report)
        
        # Mettre à jour contribution
        self.contribution.record_report(nh_id, report)
        
        # Décider si on recalcule le gradient
        if self.observation_policy.should_update_gradient(nh_id, len(self.report_buffer[nh_id])):
            self._compute_gradient(nh_id)
    
    def pull_gradient(self, nh_id: bytes) -> GradientPacket:
        """
        Retourne le gradient actuel pour un pattern.
        """
        if nh_id not in self.patterns:
            raise ValueError(f"Unknown pattern: {nh_id.hex()}")
        
        # Recalculer si nécessaire
        if nh_id not in self.gradient_cache:
            self._compute_gradient(nh_id)
        
        return self.gradient_cache[nh_id]
    
    def _compute_gradient(self, nh_id: bytes) -> None:
        """
        Calcule le gradient pour un pattern.
        """
        reports = self.report_buffer[nh_id]
        if not reports:
            return
        
        # Encoder les reports
        encoded = self._encode_reports(reports)
        
        # Obtenir les gradients de chaque critic
        critic_gradients = []
        critic_ids = []
        
        for critic_id, critic in enumerate(self.critics.critics):
            grad = critic.compute_gradient(encoded, self.horizons)
            critic_gradients.append(grad)
            critic_ids.append(critic_id.to_bytes(4, 'big'))
        
        # Agréger
        param_grad = torch.stack(critic_gradients).mean(dim=0)
        
        # Calculer incertitude
        uncertainty = self._compute_uncertainty(critic_gradients)
        
        # Calculer allocation
        contribution_score = self.contribution.get_score(nh_id)
        alloc_signal = self.tau_allocator.compute(
            nh_id, 
            param_grad, 
            uncertainty,
            contribution_score
        )
        
        # Mettre en cache
        self.gradient_cache[nh_id] = GradientPacket(
            nh_id=nh_id,
            param_grad=param_grad,
            horizons=self.horizons,
            alloc_signal=alloc_signal,
            uncertainty=uncertainty,
            critic_ids=critic_ids,
            timestamp=int(time.time() * 1000)
        )
        
        # Mettre à jour tau_rate du pattern
        self.patterns[nh_id]['tau_rate'] = alloc_signal.tau_rate
        
        # Vider le buffer (garder les derniers pour continuité)
        self.report_buffer[nh_id] = reports[-10:]
    
    def _encode_reports(self, reports: List[Report]) -> torch.Tensor:
        """
        Encode une liste de reports en tenseur.
        """
        encoded = []
        for r in reports:
            concat = torch.cat([
                r.state.flatten(),
                r.action.flatten(),
                r.cost.flatten(),
                r.outcome.flatten()
            ])
            encoded.append(concat)
        
        return torch.stack(encoded)
    
    def _compute_uncertainty(self, critic_gradients: List[torch.Tensor]) -> UncertaintyBundle:
        """
        Calcule l'incertitude structurée.
        """
        stacked = torch.stack(critic_gradients)
        
        # Epistemic = désaccord entre critics
        epistemic = stacked.var(dim=0).mean().item()
        
        # Autres = stubs pour V0
        aleatoric = 0.0
        model = 0.0
        adversarial = 0.0
        
        # Scalar = agrégation simple
        scalar = epistemic
        
        return UncertaintyBundle(
            epistemic=epistemic,
            aleatoric=aleatoric,
            model=model,
            adversarial=adversarial,
            scalar=scalar
        )
    
    def get_contribution(self, nh_id: bytes):
        """
        Retourne le score de contribution d'un pattern.
        """
        return self.contribution.get_score(nh_id)
    
    def fork(self, parent_nh_id: bytes, n_children: int = 2) -> List[bytes]:
        """
        Fork un pattern en plusieurs.
        """
        if parent_nh_id not in self.patterns:
            raise ValueError(f"Unknown pattern: {parent_nh_id.hex()}")
        
        parent = self.patterns[parent_nh_id]
        children = []
        
        for i in range(n_children):
            child_id = self.register(
                param_shape=parent['param_shape'],
                lineage=parent['lineage'] + [parent_nh_id]
            )
            children.append(child_id)
            
            # Partager tau_rate
            self.patterns[child_id]['tau_rate'] = parent['tau_rate'] / n_children
        
        # Émettre événement
        self.lifecycle.emit_fork(parent_nh_id, children)
        
        return children
    
    def freeze(self, nh_id: bytes) -> None:
        """
        Gèle un pattern (tau_rate = 0).
        """
        if nh_id not in self.patterns:
            raise ValueError(f"Unknown pattern: {nh_id.hex()}")
        
        self.patterns[nh_id]['tau_rate'] = 0.0
        self.patterns[nh_id]['status'] = 'frozen'
        self.lifecycle.emit_freeze(nh_id)
    
    def dissolve(self, nh_id: bytes) -> None:
        """
        Dissout un pattern définitivement.
        """
        if nh_id not in self.patterns:
            raise ValueError(f"Unknown pattern: {nh_id.hex()}")
        
        # Archiver dans la mémoire globale
        self.memory.archive_pattern(nh_id)
        
        # Émettre événement
        self.lifecycle.emit_dissolve(nh_id)
        
        # Supprimer
        del self.patterns[nh_id]
        if nh_id in self.report_buffer:
            del self.report_buffer[nh_id]
        if nh_id in self.gradient_cache:
            del self.gradient_cache[nh_id]


if __name__ == "__main__":
    # Test basique
    config = {
        'horizons': [1.0, 10.0],
        'n_critics': 2,
        'critic_hidden_dim': 256,
        'memory_path': './test_memory'
    }
    
    kernel = SyntheticFieldKernel(config)
    
    # Enregistrer un pattern
    nh_id = kernel.register(param_shape=(128,))
    print(f"Registered pattern: {nh_id.hex()[:16]}...")
    
    # Simuler des reports
    for i in range(10):
        report = Report(
            nh_id=nh_id,
            state=torch.randn(32),
            action=torch.randn(8),
            cost=torch.tensor([0.1]),
            outcome=torch.randn(16),
            timestamp=int(time.time() * 1000)
        )
        kernel.report(report)
    
    # Pull gradient
    grad = kernel.pull_gradient(nh_id)
    print(f"Gradient shape: {grad.param_grad.shape}")
    print(f"Horizons: {grad.horizons}")
    print(f"Tau rate: {grad.alloc_signal.tau_rate}")
    print(f"Epistemic uncertainty: {grad.uncertainty.epistemic}")
    
    print("\n✓ V0 Field Kernel operational")