"""
Contribution Tracker - Skin in the game synthétique.
"""

from dataclasses import dataclass
from typing import Dict
import time


@dataclass
class ContributionScore:
    reports_submitted: int = 0
    gradient_utility: float = 0.0
    data_uniqueness: float = 0.0
    access_level: int = 0


class ContributionTracker:
    """
    Suit les contributions des patterns.
    """
    
    def __init__(self):
        self.scores: Dict[bytes, ContributionScore] = {}
        self.report_history: Dict[bytes, list] = {}
        
        # Seuils pour access levels
        self.level_thresholds = [
            (100, 0.1, 0.1),    # Level 1
            (1000, 0.3, 0.3),   # Level 2
            (10000, 0.6, 0.6),  # Level 3
        ]
    
    def init_pattern(self, nh_id: bytes):
        """Initialise le tracking pour un nouveau pattern."""
        self.scores[nh_id] = ContributionScore()
        self.report_history[nh_id] = []
    
    def record_report(self, nh_id: bytes, report):
        """Enregistre un report."""
        if nh_id not in self.scores:
            self.init_pattern(nh_id)
        
        self.scores[nh_id].reports_submitted += 1
        self.report_history[nh_id].append({
            'timestamp': report.timestamp,
            'outcome_norm': report.outcome.norm().item()
        })
        
        # Mettre à jour access level
        self._update_access_level(nh_id)
    
    def record_gradient_usage(self, 
                              source_nh_id: bytes,
                              user_nh_id: bytes,
                              performance_delta: float):
        """
        Enregistre quand les données d'un pattern aident un autre.
        """
        if source_nh_id not in self.scores:
            return
        
        # Utility = sum des deltas de performance
        self.scores[source_nh_id].gradient_utility += max(0, performance_delta)
        self._update_access_level(source_nh_id)
    
    def update_uniqueness(self, nh_id: bytes, uniqueness: float):
        """Met à jour le score d'unicité."""
        if nh_id not in self.scores:
            return
        
        # Moyenne mobile
        alpha = 0.1
        current = self.scores[nh_id].data_uniqueness
        self.scores[nh_id].data_uniqueness = alpha * uniqueness + (1 - alpha) * current
        self._update_access_level(nh_id)
    
    def _update_access_level(self, nh_id: bytes):
        """Recalcule l'access level."""
        score = self.scores[nh_id]
        
        level = 0
        for i, (reports_thresh, utility_thresh, unique_thresh) in enumerate(self.level_thresholds):
            if (score.reports_submitted >= reports_thresh and
                score.gradient_utility >= utility_thresh and
                score.data_uniqueness >= unique_thresh):
                level = i + 1
        
        score.access_level = level
    
    def get_score(self, nh_id: bytes) -> ContributionScore:
        """Retourne le score d'un pattern."""
        if nh_id not in self.scores:
            return ContributionScore()
        return self.scores[nh_id]
    
    def get_top_contributors(self, n: int = 10) -> list:
        """Retourne les top contributeurs."""
        sorted_scores = sorted(
            self.scores.items(),
            key=lambda x: x[1].gradient_utility + x[1].data_uniqueness,
            reverse=True
        )
        return sorted_scores[:n]
    
    def apply_rewards(self):
        """Applique des récompenses aux top contributeurs."""
        top = self.get_top_contributors(n=int(len(self.scores) * 0.1) or 1)
        rewards = []
        
        for nh_id, score in top:
            # Bonus de 10% sur l'access level potentiel
            if score.access_level < 3:
                # Réduire les seuils temporairement
                rewards.append({
                    'nh_id': nh_id,
                    'type': 'access_boost',
                    'value': 0.1
                })
        
        return rewards
    
    def apply_penalties(self):
        """Applique des pénalités aux free-riders."""
        penalties = []
        
        for nh_id, score in self.scores.items():
            # Ratio contribution / consommation
            # (simplifié pour V0: basé sur reports vs utility)
            if score.reports_submitted > 100 and score.gradient_utility < 0.01:
                penalties.append({
                    'nh_id': nh_id,
                    'type': 'tau_reduction',
                    'value': 0.9  # Réduction de 10%
                })
        
        return penalties