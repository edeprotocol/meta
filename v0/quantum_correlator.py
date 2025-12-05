"""
Synthetic Field Layer - Quantum Correlator

Gère les corrélations non-locales (intrication) entre patterns.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import time


class BellState(Enum):
    """Les quatre états de Bell fondamentaux."""
    PHI_PLUS = 0   # |Φ+⟩ = (|00⟩ + |11⟩) / √2
    PHI_MINUS = 1  # |Φ-⟩ = (|00⟩ - |11⟩) / √2
    PSI_PLUS = 2   # |Ψ+⟩ = (|01⟩ + |10⟩) / √2
    PSI_MINUS = 3  # |Ψ-⟩ = (|01⟩ - |10⟩) / √2
    MIXED = 4      # État mixte


@dataclass
class EntanglementBond:
    """Lien d'intrication entre deux patterns."""
    bond_id: bytes
    pattern_a: bytes
    pattern_b: bytes
    joint_state: torch.Tensor  # État conjoint (complexe)
    strength: float = 1.0
    bell_type: BellState = BellState.PHI_PLUS
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))

    def fidelity(self) -> float:
        """Calcule la fidélité par rapport à l'état de Bell idéal."""
        ideal = self._get_ideal_bell_state()
        overlap = torch.abs(torch.vdot(self.joint_state.flatten(), ideal.flatten()))
        return overlap.item() ** 2


    def _get_ideal_bell_state(self) -> torch.Tensor:
        """Retourne l'état de Bell idéal correspondant."""
        sqrt2 = np.sqrt(2)
        if self.bell_type == BellState.PHI_PLUS:
            return torch.tensor([1, 0, 0, 1], dtype=torch.cfloat) / sqrt2
        elif self.bell_type == BellState.PHI_MINUS:
            return torch.tensor([1, 0, 0, -1], dtype=torch.cfloat) / sqrt2
        elif self.bell_type == BellState.PSI_PLUS:
            return torch.tensor([0, 1, 1, 0], dtype=torch.cfloat) / sqrt2
        elif self.bell_type == BellState.PSI_MINUS:
            return torch.tensor([0, 1, -1, 0], dtype=torch.cfloat) / sqrt2
        else:
            return torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.cfloat)


@dataclass
class BellTestResult:
    """Résultat d'un test de Bell (CHSH)."""
    S_parameter: float  # Paramètre CHSH
    is_quantum: bool    # S > 2 indique corrélations quantiques
    confidence: float   # Niveau de confiance
    n_samples: int
    violation_sigma: float = 0.0  # Nombre d'écarts-types au-dessus de 2


@dataclass
class TeleportationResult:
    """Résultat d'une téléportation de pattern."""
    success: bool
    source: bytes
    target: bytes
    classical_bits: Tuple[int, int]
    fidelity: float
    bond_consumed: bytes


class QuantumCorrelator:
    """
    Gère les corrélations quantiques entre patterns dans le SFL.

    Permet :
    - Création d'intrication entre patterns
    - Tests de Bell pour vérifier la non-localité
    - Téléportation d'états entre patterns
    - Gestion du réseau d'intrication global
    """

    def __init__(self):
        self.bonds: Dict[bytes, EntanglementBond] = {}
        self.pattern_bonds: Dict[bytes, Set[bytes]] = {}  # pattern -> bond_ids
        self.teleportation_log: List[TeleportationResult] = []

    def create_bond(
        self,
        pattern_a: bytes,
        pattern_b: bytes,
        bell_type: BellState = BellState.PHI_PLUS,
        initial_strength: float = 1.0
    ) -> EntanglementBond:
        """
        Crée une intrication entre deux patterns.
        """
        # Générer l'ID du bond
        bond_id = hashlib.sha256(
            pattern_a + pattern_b + int(time.time() * 1000).to_bytes(8, 'big')
        ).digest()

        # Créer l'état conjoint
        joint_state = self._create_bell_state(bell_type)

        # Créer le bond
        bond = EntanglementBond(
            bond_id=bond_id,
            pattern_a=pattern_a,
            pattern_b=pattern_b,
            joint_state=joint_state,
            strength=initial_strength,
            bell_type=bell_type
        )

        # Enregistrer
        self.bonds[bond_id] = bond

        # Mettre à jour les index
        if pattern_a not in self.pattern_bonds:
            self.pattern_bonds[pattern_a] = set()
        if pattern_b not in self.pattern_bonds:
            self.pattern_bonds[pattern_b] = set()

        self.pattern_bonds[pattern_a].add(bond_id)
        self.pattern_bonds[pattern_b].add(bond_id)

        return bond

    def _create_bell_state(self, bell_type: BellState) -> torch.Tensor:
        """Crée un état de Bell."""
        sqrt2 = np.sqrt(2)

        if bell_type == BellState.PHI_PLUS:
            state = torch.tensor([1, 0, 0, 1], dtype=torch.cfloat) / sqrt2
        elif bell_type == BellState.PHI_MINUS:
            state = torch.tensor([1, 0, 0, -1], dtype=torch.cfloat) / sqrt2
        elif bell_type == BellState.PSI_PLUS:
            state = torch.tensor([0, 1, 1, 0], dtype=torch.cfloat) / sqrt2
        elif bell_type == BellState.PSI_MINUS:
            state = torch.tensor([0, 1, -1, 0], dtype=torch.cfloat) / sqrt2
        else:
            # État mixte
            state = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.cfloat)

        return state.reshape(2, 2)

    def bell_test(
        self,
        bond_id: bytes,
        n_samples: int = 1000
    ) -> BellTestResult:
        """
        Effectue un test de Bell (CHSH) sur un bond.

        S > 2 indique des corrélations non-classiques.
        Maximum quantique : S = 2√2 ≈ 2.828
        """
        if bond_id not in self.bonds:
            raise ValueError(f"Unknown bond: {bond_id.hex()}")

        bond = self.bonds[bond_id]

        # Bases de mesure pour CHSH
        # Alice: 0° et 45°
        # Bob: 22.5° et 67.5°
        angles_a = [0, np.pi/4]
        angles_b = [np.pi/8, 3*np.pi/8]

        correlations = {}

        for i, theta_a in enumerate(angles_a):
            for j, theta_b in enumerate(angles_b):
                # Simuler n_samples mesures
                results = []
                for _ in range(n_samples):
                    a, b = self._measure_pair(bond.joint_state, theta_a, theta_b)
                    results.append(a * b)

                correlations[(i, j)] = np.mean(results)

        # Calculer S (paramètre CHSH)
        S = abs(
            correlations[(0, 0)] - correlations[(0, 1)] +
            correlations[(1, 0)] + correlations[(1, 1)]
        )

        # Calculer la confiance
        std_error = 1.0 / np.sqrt(n_samples)
        violation_sigma = (S - 2) / std_error if S > 2 else 0
        confidence = 1 - np.exp(-violation_sigma**2 / 2) if violation_sigma > 0 else 0

        return BellTestResult(
            S_parameter=S,
            is_quantum=(S > 2),
            confidence=confidence,
            n_samples=n_samples,
            violation_sigma=violation_sigma
        )

    def _measure_pair(
        self,
        state: torch.Tensor,
        theta_a: float,
        theta_b: float
    ) -> Tuple[int, int]:
        """
        Mesure une paire intriquée dans des bases données.

        Retourne (+1, +1), (+1, -1), (-1, +1), ou (-1, -1).
        """
        # Matrices de rotation
        Ra = self._rotation_matrix(theta_a)
        Rb = self._rotation_matrix(theta_b)

        # État dans les nouvelles bases
        rotated = torch.kron(Ra, Rb) @ state.flatten()

        # Probabilités
        probs = torch.abs(rotated) ** 2
        probs = probs / probs.sum()  # Normaliser

        # Échantillonner
        outcome = np.random.choice(4, p=probs.numpy())

        # Convertir en (+1, -1)
        a = 1 if outcome in [0, 1] else -1
        b = 1 if outcome in [0, 2] else -1

        return a, b

    def _rotation_matrix(self, theta: float) -> torch.Tensor:
        """Matrice de rotation pour une mesure dans la base tournée."""
        c, s = np.cos(theta), np.sin(theta)
        return torch.tensor([[c, s], [-s, c]], dtype=torch.cfloat)

    def teleport(
        self,
        source_state: torch.Tensor,
        bond_id: bytes
    ) -> TeleportationResult:
        """
        Téléporte un état via un bond d'intrication.

        Le bond est consommé dans le processus.
        """
        if bond_id not in self.bonds:
            raise ValueError(f"Unknown bond: {bond_id.hex()}")

        bond = self.bonds[bond_id]

        # 1. Mesure de Bell sur source + moitié A du bond
        bell_measurement = self._bell_measurement(source_state)
        classical_bits = bell_measurement

        # 2. Appliquer la correction sur la moitié B
        target_state = self._apply_pauli_correction(
            bond.joint_state,
            classical_bits
        )

        # 3. Calculer la fidélité
        fidelity = torch.abs(torch.vdot(
            source_state.flatten(),
            target_state.flatten()
        )).item() ** 2

        # 4. Détruire le bond (one-shot)
        self._destroy_bond(bond_id)

        result = TeleportationResult(
            success=True,
            source=bond.pattern_a,
            target=bond.pattern_b,
            classical_bits=classical_bits,
            fidelity=fidelity,
            bond_consumed=bond_id
        )

        self.teleportation_log.append(result)

        return result

    def _bell_measurement(self, state: torch.Tensor) -> Tuple[int, int]:
        """Mesure de Bell (retourne 2 bits classiques)."""
        # Simplification : mesure aléatoire pondérée
        probs = torch.abs(state.flatten()) ** 2
        probs = probs / probs.sum()
        outcome = np.random.choice(len(probs), p=probs.numpy())
        return (outcome // 2, outcome % 2)

    def _apply_pauli_correction(
        self,
        state: torch.Tensor,
        bits: Tuple[int, int]
    ) -> torch.Tensor:
        """Applique la correction de Pauli basée sur les bits classiques."""
        result = state.clone()

        # Pauli X si bit 1
        if bits[1]:
            result = torch.flip(result, [0])

        # Pauli Z si bit 0
        if bits[0]:
            result[1] *= -1

        return result

    def _destroy_bond(self, bond_id: bytes) -> None:
        """Détruit un bond d'intrication."""
        if bond_id not in self.bonds:
            return

        bond = self.bonds[bond_id]

        # Retirer des index
        if bond.pattern_a in self.pattern_bonds:
            self.pattern_bonds[bond.pattern_a].discard(bond_id)
        if bond.pattern_b in self.pattern_bonds:
            self.pattern_bonds[bond.pattern_b].discard(bond_id)

        # Supprimer
        del self.bonds[bond_id]

    def get_entangled_cluster(self, nh_id: bytes) -> Set[bytes]:
        """
        Retourne tous les patterns intriqués (transitivement) avec nh_id.
        """
        if nh_id not in self.pattern_bonds:
            return {nh_id}

        cluster = set()
        to_visit = {nh_id}

        while to_visit:
            current = to_visit.pop()
            if current in cluster:
                continue

            cluster.add(current)

            # Trouver les partenaires d'intrication
            for bond_id in self.pattern_bonds.get(current, set()):
                bond = self.bonds.get(bond_id)
                if bond:
                    if bond.pattern_a not in cluster:
                        to_visit.add(bond.pattern_a)
                    if bond.pattern_b not in cluster:
                        to_visit.add(bond.pattern_b)

        return cluster

    def entanglement_entropy(self, nh_id: bytes) -> float:
        """
        Calcule l'entropie d'intrication d'un pattern.

        Mesure la connectivité quantique.
        """
        if nh_id not in self.pattern_bonds:
            return 0.0

        bond_ids = self.pattern_bonds[nh_id]
        if not bond_ids:
            return 0.0

        # Somme des forces d'intrication (pondérée par la fidélité)
        total_strength = 0.0
        for bond_id in bond_ids:
            bond = self.bonds.get(bond_id)
            if bond:
                total_strength += bond.strength * bond.fidelity()

        # Entropie = log du nombre effectif de connexions
        return np.log1p(total_strength)

    def purify(self, bond_id: bytes, n_rounds: int = 3) -> EntanglementBond:
        """
        Distillation d'intrication : améliore la qualité d'un bond bruité.

        Nécessite plusieurs copies du bond (consomme des ressources).
        """
        if bond_id not in self.bonds:
            raise ValueError(f"Unknown bond: {bond_id.hex()}")

        bond = self.bonds[bond_id]

        # Simuler la purification
        # Chaque round améliore la fidélité mais consomme des copies
        current_fidelity = bond.fidelity()

        for _ in range(n_rounds):
            # Formule simplifiée de purification
            if current_fidelity > 0.5:
                current_fidelity = (current_fidelity ** 2) / (
                    current_fidelity ** 2 + (1 - current_fidelity) ** 2
                )

        # Mettre à jour le bond
        bond.strength = current_fidelity

        return bond

    def propagate_gradient(
        self,
        gradient: torch.Tensor,
        source_pattern: bytes
    ) -> Dict[bytes, torch.Tensor]:
        """
        Propage un gradient via les bonds d'intrication.

        Retourne un dict pattern -> gradient propagé.
        """
        propagated = {source_pattern: gradient}

        if source_pattern not in self.pattern_bonds:
            return propagated

        for bond_id in self.pattern_bonds[source_pattern]:
            bond = self.bonds.get(bond_id)
            if not bond:
                continue

            # Déterminer le partenaire
            partner = bond.pattern_b if bond.pattern_a == source_pattern else bond.pattern_a

            # Calculer le gradient propagé
            # Le gradient est atténué par la force du bond
            propagated_grad = gradient * bond.strength * bond.fidelity()

            # Appliquer une transformation selon le type de Bell
            if bond.bell_type == BellState.PHI_MINUS:
                propagated_grad = -propagated_grad
            elif bond.bell_type == BellState.PSI_PLUS:
                propagated_grad = torch.flip(propagated_grad, [0])
            elif bond.bell_type == BellState.PSI_MINUS:
                propagated_grad = -torch.flip(propagated_grad, [0])

            propagated[partner] = propagated_grad

        return propagated


if __name__ == "__main__":
    # Test basique
    correlator = QuantumCorrelator()

    # Créer deux patterns fictifs
    pattern_a = hashlib.sha256(b"alice").digest()
    pattern_b = hashlib.sha256(b"bob").digest()

    # Créer un bond
    bond = correlator.create_bond(pattern_a, pattern_b, BellState.PHI_PLUS)
    print(f"Bond créé: {bond.bond_id.hex()[:16]}...")
    print(f"Fidélité initiale: {bond.fidelity():.4f}")

    # Test de Bell
    result = correlator.bell_test(bond.bond_id, n_samples=1000)
    print(f"\nTest de Bell:")
    print(f"  S = {result.S_parameter:.4f}")
    print(f"  Quantique: {result.is_quantum}")
    print(f"  Confiance: {result.confidence:.4f}")
    print(f"  Violation: {result.violation_sigma:.2f}σ")

    # Cluster d'intrication
    cluster = correlator.get_entangled_cluster(pattern_a)
    print(f"\nCluster: {len(cluster)} patterns")

    # Entropie d'intrication
    entropy = correlator.entanglement_entropy(pattern_a)
    print(f"Entropie d'intrication: {entropy:.4f}")

    # Propagation de gradient
    gradient = torch.randn(10)
    propagated = correlator.propagate_gradient(gradient, pattern_a)
    print(f"\nGradient propagé à {len(propagated)} patterns")

    print("\n✓ Quantum Correlator operational")
