"""
Observation Policy - Décide quoi observer et quand.
"""

import random
from typing import Dict
import time


class ObservationPolicy:
    """
    Politique d'observation.
    
    En V0: simple mais intentionnel.
    """
    
    def __init__(self):
        # Paramètres
        self.base_update_interval = 10  # Reports avant update gradient
        self.tau_weight = 0.5           # Poids de tau_rate
        self.uncertainty_weight = 0.3   # Poids de l'incertitude
        self.contribution_weight = 0.2  # Poids de la contribution
        
        # État
        self.last_update: Dict[bytes, int] = {}
        self.pattern_stats: Dict[bytes, dict] = {}
    
    def should_update_gradient(self, 
                               nh_id: bytes, 
                               buffer_size: int) -> bool:
        """
        Décide si on doit recalculer le gradient pour un pattern.
        """
        # Initialiser si nouveau
        if nh_id not in self.last_update:
            self.last_update[nh_id] = 0
            self.pattern_stats[nh_id] = {
                'tau_rate': 1.0,
                'uncertainty': 0.5,
                'contribution': 0.0
            }
        
        # Calculer l'intervalle adaptatif
        stats = self.pattern_stats[nh_id]
        
        # Plus tau_rate est haut, plus on observe souvent
        tau_factor = 1.0 / (stats['tau_rate'] + 0.1)
        
        # Plus l'incertitude est haute, plus on observe
        uncertainty_factor = 1.0 / (stats['uncertainty'] + 0.1)
        
        # Plus la contribution est haute, plus on observe
        contribution_factor = 1.0 / (stats['contribution'] + 0.1)
        
        adaptive_interval = self.base_update_interval * (
            self.tau_weight * tau_factor +
            self.uncertainty_weight * uncertainty_factor +
            self.contribution_weight * contribution_factor
        )
        
        # Minimum 1, maximum 100
        adaptive_interval = max(1, min(100, int(adaptive_interval)))
        
        # Décision
        should_update = buffer_size >= adaptive_interval
        
        if should_update:
            self.last_update[nh_id] = buffer_size
        
        return should_update
    
    def update_stats(self, 
                     nh_id: bytes,
                     tau_rate: float,
                     uncertainty: float,
                     contribution: float):
        """Met à jour les stats d'un pattern."""
        if nh_id not in self.pattern_stats:
            self.pattern_stats[nh_id] = {}
        
        self.pattern_stats[nh_id] = {
            'tau_rate': tau_rate,
            'uncertainty': uncertainty,
            'contribution': contribution
        }
    
    def get_observation_priority(self, nh_id: bytes) -> float:
        """
        Retourne la priorité d'observation d'un pattern.
        
        Utilisé pour décider quels patterns observer en premier
        quand les ressources sont limitées.
        """
        if nh_id not in self.pattern_stats:
            return 1.0  # Default
        
        stats = self.pattern_stats[nh_id]
        
        priority = (
            self.tau_weight * stats['tau_rate'] +
            self.uncertainty_weight * stats['uncertainty'] +
            self.contribution_weight * (1.0 + stats['contribution'])
        )
        
        return priority