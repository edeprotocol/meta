"""
Ensemble de Critics pour le Synthetic Field Layer.
"""

import torch
import torch.nn as nn
from typing import List


class Critic(nn.Module):
    """
    Un critic qui estime la valeur et produit des gradients.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, n_horizons: int):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Une tête par horizon
        self.value_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(n_horizons)
        ])
        
        self.n_horizons = n_horizons
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Retourne les valeurs pour chaque horizon.
        """
        encoded = self.encoder(x)
        values = torch.stack([head(encoded) for head in self.value_heads], dim=-1)
        return values.squeeze(-2)  # [batch, n_horizons]
    
    def compute_gradient(self, 
                         reports_encoded: torch.Tensor,
                         horizons: List[float]) -> torch.Tensor:
        """
        Calcule le gradient pour les paramètres.
        
        Simplifié pour V0: gradient = direction qui augmente la valeur.
        """
        reports_encoded = reports_encoded.requires_grad_(True)
        
        values = self.forward(reports_encoded)
        
        # Pondérer par horizon (long terme pèse plus)
        weights = torch.tensor([1.0 / h for h in horizons])
        weighted_value = (values * weights).sum()
        
        # Gradient par rapport aux inputs
        weighted_value.backward()
        
        grad = reports_encoded.grad.mean(dim=0)  # Moyenne sur le batch
        
        # Étendre aux dimensions [d_p, H]
        # Pour V0, on répète le même gradient pour chaque horizon avec scaling
        d_p = grad.shape[0]
        param_grad = torch.stack([grad * w for w in weights], dim=1)
        
        return param_grad


class CriticEnsemble:
    """
    Ensemble de critics indépendants.
    """
    
    def __init__(self, n_critics: int, n_horizons: int, hidden_dim: int):
        self.n_critics = n_critics
        self.n_horizons = n_horizons
        self.hidden_dim = hidden_dim
        
        # Dimension d'input sera déterminée dynamiquement
        self.critics: List[Critic] = []
        self.initialized = False
    
    def _init_critics(self, input_dim: int):
        """
        Initialise les critics avec la bonne dimension d'input.
        """
        self.critics = [
            Critic(input_dim, self.hidden_dim, self.n_horizons)
            for _ in range(self.n_critics)
        ]
        self.initialized = True
    
    def compute_gradients(self, 
                          reports_encoded: torch.Tensor,
                          horizons: List[float]) -> List[torch.Tensor]:
        """
        Calcule les gradients de tous les critics.
        """
        if not self.initialized:
            self._init_critics(reports_encoded.shape[-1])
        
        gradients = []
        for critic in self.critics:
            grad = critic.compute_gradient(reports_encoded.clone(), horizons)
            gradients.append(grad)
        
        return gradients
    
    def get_epistemic_uncertainty(self, gradients: List[torch.Tensor]) -> float:
        """
        Calcule le désaccord entre critics.
        """
        stacked = torch.stack(gradients)
        variance = stacked.var(dim=0).mean().item()
        return variance