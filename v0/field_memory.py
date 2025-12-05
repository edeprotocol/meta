"""
Field Memory - L'intégrale du champ.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
import pyarrow as pa
import pyarrow.parquet as pq
from dataclasses import dataclass
import time


@dataclass
class MemoryEntry:
    nh_id: bytes
    embedding: np.ndarray
    weight: float
    timestamp: int


class FieldMemory:
    """
    Mémoire du champ - intégrale temporelle de Φ.
    """
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Intégrale courante (en mémoire)
        self.integral: Optional[torch.Tensor] = None
        self.total_weight: float = 0.0
        
        # Buffer pour écriture batch
        self.buffer: List[MemoryEntry] = []
        self.buffer_size = 1000
        
        # Index des contributions par pattern
        self.contribution_index: Dict[bytes, float] = {}
        
        # Encoder simple (MLP)
        self.encoder = None
        self.encoder_dim = 128
    
    def _init_encoder(self, input_dim: int):
        """Initialise l'encodeur."""
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.encoder_dim)
        )
    
    def _encode(self, report) -> torch.Tensor:
        """Encode un report en embedding."""
        concat = torch.cat([
            report.state.flatten(),
            report.action.flatten(),
            report.cost.flatten(),
            report.outcome.flatten()
        ])
        
        if self.encoder is None:
            self._init_encoder(concat.shape[0])
        
        with torch.no_grad():
            embedding = self.encoder(concat)
        
        return embedding
    
    def _compute_weight(self, report, embedding: torch.Tensor) -> float:
        """Calcule le poids d'un report."""
        # Recency
        age = (time.time() * 1000 - report.timestamp) / 1000  # en secondes
        w_recency = np.exp(-0.001 * age)
        
        # Uniqueness (distance à l'intégrale actuelle)
        if self.integral is not None:
            distance = torch.norm(embedding - self.integral / (self.total_weight + 1e-6))
            w_uniqueness = torch.sigmoid(distance).item()
        else:
            w_uniqueness = 1.0
        
        return w_recency * w_uniqueness
    
    def integrate(self, report) -> None:
        """Intègre un report dans la mémoire."""
        embedding = self._encode(report)
        weight = self._compute_weight(report, embedding)
        
        # Mettre à jour l'intégrale
        if self.integral is None:
            self.integral = embedding * weight
        else:
            self.integral = self.integral + embedding * weight
        
        self.total_weight += weight
        
        # Tracker contribution
        nh_id = report.nh_id
        if nh_id not in self.contribution_index:
            self.contribution_index[nh_id] = 0.0
        self.contribution_index[nh_id] += weight
        
        # Ajouter au buffer
        self.buffer.append(MemoryEntry(
            nh_id=nh_id,
            embedding=embedding.detach().numpy(),
            weight=weight,
            timestamp=report.timestamp
        ))
        
        # Flush si nécessaire
        if len(self.buffer) >= self.buffer_size:
            self._flush_buffer()
    
    def _flush_buffer(self):
        """Écrit le buffer sur disque."""
        if not self.buffer:
            return
        
        # Convertir en Arrow table
        table = pa.table({
            'nh_id': [e.nh_id for e in self.buffer],
            'embedding': [e.embedding.tobytes() for e in self.buffer],
            'weight': [e.weight for e in self.buffer],
            'timestamp': [e.timestamp for e in self.buffer]
        })
        
        # Écrire en Parquet
        filename = f"memory_{int(time.time() * 1000)}.parquet"
        pq.write_table(table, self.storage_path / filename)
        
        # Vider le buffer
        self.buffer = []
    
    def influence_on_gradient(self, nh_id: bytes) -> torch.Tensor:
        """
        Calcule comment l'historique influence le gradient d'un pattern.
        """
        if self.integral is None:
            return torch.zeros(self.encoder_dim)
        
        # Contribution relative de ce pattern
        pattern_contribution = self.contribution_index.get(nh_id, 0.0)
        relative_contribution = pattern_contribution / (self.total_weight + 1e-6)
        
        # Influence = projection pondérée
        influence = self.integral * relative_contribution / (self.total_weight + 1e-6)
        
        return influence
    
    def query(self, 
              projection: torch.Tensor,
              time_range: Optional[tuple] = None) -> torch.Tensor:
        """Projette l'intégrale sur une base."""
        if self.integral is None:
            return torch.zeros(projection.shape[0])
        
        normalized_integral = self.integral / (self.total_weight + 1e-6)
        return projection @ normalized_integral
    
    def archive_pattern(self, nh_id: bytes):
        """Archive un pattern dissous."""
        # La contribution reste dans l'intégrale
        # On marque juste le pattern comme archivé
        if nh_id in self.contribution_index:
            contribution = self.contribution_index[nh_id]
            # Log pour audit
            print(f"Archived pattern {nh_id.hex()[:16]} with contribution {contribution}")
    
    def get_stats(self) -> dict:
        """Retourne des statistiques sur la mémoire."""
        return {
            'total_weight': self.total_weight,
            'n_patterns': len(self.contribution_index),
            'buffer_size': len(self.buffer),
            'integral_norm': torch.norm(self.integral).item() if self.integral is not None else 0.0
        }