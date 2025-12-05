"""
Tau Allocator - Distribution du temps propre.
"""

import torch
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class AllocSignal:
    tau_rate: float = 1.0
    allowed_envs: list = None
    compliance_window: int = 100
    non_compliance_penalty: float = 0.1
    
    def __post_init__(self):
        if self.allowed_envs is None:
            self.allowed_envs = []


class TauAllocator:
    """
    Alloue le temps propre aux patterns.
    
    tau_rate détermine la "vitesse" d'un pattern dans le champ.
    """
    
    def __init__(self):
        # Paramètres
        self.base_tau = 1.0
        self.min_tau = 0.01
        self.max_tau = 10.0
        
        # Poids des facteurs
        self.performance_weight = 0.4
        self.contribution_weight = 0.3
        self.uncertainty_weight = 0.2
        self.stability_weight = 0.1
        
        # Historique pour stabilité
        self.tau_history: Dict[bytes, list] = {}
        self.gradient_history: Dict[bytes, list] = {}
    
    def compute(self,
                nh_id: bytes,
                param_grad: torch.Tensor,
                uncertainty,  # UncertaintyBundle
                contribution  # ContributionScore
                ) -> AllocSignal:
        """
        Calcule le signal d'allocation pour un pattern.
        """
        # Performance = magnitude du gradient (plus grand = plus d'impact)
        grad_magnitude = param_grad.norm().item()
        performance_score = self._normalize(grad_magnitude, 0, 10)
        
        # Contribution
        contribution_score = self._normalize(
            contribution.gradient_utility + contribution.data_uniqueness,
            0, 2
        )
        
        # Incertitude (haute incertitude = plus de tau pour explorer)
        uncertainty_score = uncertainty.scalar
        
        # Stabilité (variance du tau historique)
        stability_score = self._compute_stability(nh_id)
        
        # Combiner
        raw_tau = (
            self.performance_weight * performance_score +
            self.contribution_weight * contribution_score +
            self.uncertainty_weight * uncertainty_score +
            self.stability_weight * stability_score
        )
        
        # Normaliser et borner
        tau_rate = self.base_tau * (0.5 + raw_tau)
        tau_rate = max(self.min_tau, min(self.max_tau, tau_rate))
        
        # Mettre à jour l'historique
        if nh_id not in self.tau_history:
            self.tau_history[nh_id] = []
        self.tau_history[nh_id].append(tau_rate)
        if len(self.tau_history[nh_id]) > 100:
            self.tau_history[nh_id] = self.tau_history[nh_id][-100:]
        
        # Stocker gradient pour stabilité
        if nh_id not in self.gradient_history:
            self.gradient_history[nh_id] = []
        self.gradient_history[nh_id].append(grad_magnitude)
        if len(self.gradient_history[nh_id]) > 100:
            self.gradient_history[nh_id] = self.gradient_history[nh_id][-100:]
        
        return AllocSignal(
            tau_rate=tau_rate,
            allowed_envs=[],  # Tous les envs par défaut
            compliance_window=100,
            non_compliance_penalty=0.1
        )
    
    def _normalize(self, value: float, min_val: float, max_val: float) -> float:
        """Normalise entre 0 et 1."""
        if max_val == min_val:
            return 0.5
        return (value - min_val) / (max_val - min_val)
    
    def _compute_stability(self, nh_id: bytes) -> float:
        """
        Calcule un score de stabilité basé sur l'historique.
        
        Patterns stables → tau plus prévisible
        Patterns instables → besoin de plus d'observation
        """
        if nh_id not in self.gradient_history or len(self.gradient_history[nh_id]) < 2:
            return 0.5  # Default pour nouveaux patterns
        
        history = self.gradient_history[nh_id]
        variance = sum((x - sum(history)/len(history))**2 for x in history) / len(history)
        
        # Inverse: haute variance = basse stabilité
        stability = 1.0 / (1.0 + variance)
        
        return stability
    
    def freeze(self, nh_id: bytes) -> AllocSignal:
        """Retourne un signal de gel (tau = 0)."""
        return AllocSignal(
            tau_rate=0.0,
            allowed_envs=[],
            compliance_window=0,
            non_compliance_penalty=0.0
        )
    
    def boost(self, nh_id: bytes, factor: float = 1.5) -> Optional[AllocSignal]:
        """Boost temporaire du tau_rate."""
        if nh_id not in self.tau_history or not self.tau_history[nh_id]:
            return None
        
        current_tau = self.tau_history[nh_id][-1]
        boosted_tau = min(self.max_tau, current_tau * factor)
        
        return AllocSignal(
            tau_rate=boosted_tau,
            allowed_envs=[],
            compliance_window=100,
            non_compliance_penalty=0.1
        )