"""
Quantum Correlator - Non-local correlations between patterns.

Implements computational entanglement for the Synthetic Field.
"""

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch


class BellState(Enum):
    """The four fundamental Bell states."""
    PHI_PLUS = 0   # |Φ+⟩ = (|00⟩ + |11⟩) / √2  - Pure cooperation
    PHI_MINUS = 1  # |Φ-⟩ = (|00⟩ - |11⟩) / √2  - Cooperation with phase
    PSI_PLUS = 2   # |Ψ+⟩ = (|01⟩ + |10⟩) / √2  - Constructive competition
    PSI_MINUS = 3  # |Ψ-⟩ = (|01⟩ - |10⟩) / √2  - Destructive competition
    MIXED = 4      # Mixed state


@dataclass
class EntanglementBond:
    """Entanglement bond between two patterns."""
    bond_id: bytes
    pattern_a: bytes
    pattern_b: bytes
    joint_state: torch.Tensor  # Complex joint state
    strength: float = 1.0
    bell_type: BellState = BellState.PHI_PLUS
    created_at: int = field(default_factory=lambda: int(time.time() * 1000))

    def fidelity(self) -> float:
        """Calculate fidelity with respect to ideal Bell state."""
        ideal = self._get_ideal_bell_state()
        overlap = torch.abs(torch.vdot(self.joint_state.flatten(), ideal.flatten()))
        return overlap.item() ** 2

    def _get_ideal_bell_state(self) -> torch.Tensor:
        """Get the ideal Bell state for this bond type."""
        sqrt2 = np.sqrt(2)
        states = {
            BellState.PHI_PLUS: torch.tensor([1, 0, 0, 1], dtype=torch.cfloat) / sqrt2,
            BellState.PHI_MINUS: torch.tensor([1, 0, 0, -1], dtype=torch.cfloat) / sqrt2,
            BellState.PSI_PLUS: torch.tensor([0, 1, 1, 0], dtype=torch.cfloat) / sqrt2,
            BellState.PSI_MINUS: torch.tensor([0, 1, -1, 0], dtype=torch.cfloat) / sqrt2,
            BellState.MIXED: torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.cfloat),
        }
        return states.get(self.bell_type, states[BellState.MIXED])


@dataclass
class BellTestResult:
    """Result of a Bell (CHSH) test."""
    S_parameter: float  # CHSH parameter
    is_quantum: bool    # S > 2 indicates quantum correlations
    confidence: float   # Confidence level
    n_samples: int
    violation_sigma: float = 0.0  # Standard deviations above 2


@dataclass
class TeleportationResult:
    """Result of a pattern teleportation."""
    success: bool
    source: bytes
    target: bytes
    classical_bits: Tuple[int, int]
    fidelity: float
    bond_consumed: bytes


class QuantumCorrelator:
    """
    Manages quantum correlations between patterns in the SFL.

    Enables:
    - Entanglement creation between patterns
    - Bell tests to verify non-locality
    - State teleportation via entanglement
    - Global entanglement network management
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
        initial_strength: float = 1.0,
    ) -> EntanglementBond:
        """
        Create an entanglement bond between two patterns.

        Args:
            pattern_a: First pattern ID
            pattern_b: Second pattern ID
            bell_type: Type of Bell state
            initial_strength: Initial bond strength

        Returns:
            The created EntanglementBond
        """
        # Generate bond ID
        bond_id = hashlib.sha256(
            pattern_a + pattern_b + int(time.time() * 1000).to_bytes(8, "big")
        ).digest()

        # Create joint state
        joint_state = self._create_bell_state(bell_type)

        bond = EntanglementBond(
            bond_id=bond_id,
            pattern_a=pattern_a,
            pattern_b=pattern_b,
            joint_state=joint_state,
            strength=initial_strength,
            bell_type=bell_type,
        )

        # Register
        self.bonds[bond_id] = bond

        # Update indices
        for pattern in [pattern_a, pattern_b]:
            if pattern not in self.pattern_bonds:
                self.pattern_bonds[pattern] = set()
            self.pattern_bonds[pattern].add(bond_id)

        return bond

    def _create_bell_state(self, bell_type: BellState) -> torch.Tensor:
        """Create a Bell state tensor."""
        sqrt2 = np.sqrt(2)
        states = {
            BellState.PHI_PLUS: torch.tensor([1, 0, 0, 1], dtype=torch.cfloat) / sqrt2,
            BellState.PHI_MINUS: torch.tensor([1, 0, 0, -1], dtype=torch.cfloat) / sqrt2,
            BellState.PSI_PLUS: torch.tensor([0, 1, 1, 0], dtype=torch.cfloat) / sqrt2,
            BellState.PSI_MINUS: torch.tensor([0, 1, -1, 0], dtype=torch.cfloat) / sqrt2,
            BellState.MIXED: torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=torch.cfloat),
        }
        return states.get(bell_type, states[BellState.MIXED]).reshape(2, 2)

    def bell_test(self, bond_id: bytes, n_samples: int = 1000) -> BellTestResult:
        """
        Perform a Bell (CHSH) test on a bond.

        S > 2 indicates non-classical correlations.
        Maximum quantum value: S = 2√2 ≈ 2.828

        Args:
            bond_id: Bond to test
            n_samples: Number of measurement samples

        Returns:
            BellTestResult with S parameter and analysis
        """
        if bond_id not in self.bonds:
            raise ValueError(f"Unknown bond: {bond_id.hex()}")

        bond = self.bonds[bond_id]

        # CHSH measurement angles
        # Alice: 0° and 45°
        # Bob: 22.5° and 67.5°
        angles_a = [0, np.pi / 4]
        angles_b = [np.pi / 8, 3 * np.pi / 8]

        correlations = {}

        for i, theta_a in enumerate(angles_a):
            for j, theta_b in enumerate(angles_b):
                results = []
                for _ in range(n_samples):
                    a, b = self._measure_pair(bond.joint_state, theta_a, theta_b)
                    results.append(a * b)
                correlations[(i, j)] = np.mean(results)

        # Calculate S (CHSH parameter)
        S = abs(
            correlations[(0, 0)]
            - correlations[(0, 1)]
            + correlations[(1, 0)]
            + correlations[(1, 1)]
        )

        # Calculate confidence
        std_error = 1.0 / np.sqrt(n_samples)
        violation_sigma = (S - 2) / std_error if S > 2 else 0
        confidence = 1 - np.exp(-violation_sigma**2 / 2) if violation_sigma > 0 else 0

        return BellTestResult(
            S_parameter=S,
            is_quantum=(S > 2),
            confidence=confidence,
            n_samples=n_samples,
            violation_sigma=violation_sigma,
        )

    def _measure_pair(
        self, state: torch.Tensor, theta_a: float, theta_b: float
    ) -> Tuple[int, int]:
        """Measure an entangled pair in given bases."""
        Ra = self._rotation_matrix(theta_a)
        Rb = self._rotation_matrix(theta_b)

        rotated = torch.kron(Ra, Rb) @ state.flatten()
        probs = torch.abs(rotated) ** 2
        probs = probs / probs.sum()

        outcome = np.random.choice(4, p=probs.numpy())
        a = 1 if outcome in [0, 1] else -1
        b = 1 if outcome in [0, 2] else -1

        return a, b

    def _rotation_matrix(self, theta: float) -> torch.Tensor:
        """Rotation matrix for measurement in rotated basis."""
        c, s = np.cos(theta), np.sin(theta)
        return torch.tensor([[c, s], [-s, c]], dtype=torch.cfloat)

    def teleport(
        self, source_state: torch.Tensor, bond_id: bytes
    ) -> TeleportationResult:
        """
        Teleport a state via an entanglement bond.

        The bond is consumed in the process (one-shot).

        Args:
            source_state: State to teleport
            bond_id: Entanglement bond to use

        Returns:
            TeleportationResult with outcome
        """
        if bond_id not in self.bonds:
            raise ValueError(f"Unknown bond: {bond_id.hex()}")

        bond = self.bonds[bond_id]

        # Bell measurement
        bell_measurement = self._bell_measurement(source_state)
        classical_bits = bell_measurement

        # Apply Pauli correction
        target_state = self._apply_pauli_correction(bond.joint_state, classical_bits)

        # Calculate fidelity
        fidelity = (
            torch.abs(torch.vdot(source_state.flatten(), target_state.flatten())).item()
            ** 2
        )

        # Destroy bond (one-shot)
        self._destroy_bond(bond_id)

        result = TeleportationResult(
            success=True,
            source=bond.pattern_a,
            target=bond.pattern_b,
            classical_bits=classical_bits,
            fidelity=fidelity,
            bond_consumed=bond_id,
        )
        self.teleportation_log.append(result)

        return result

    def _bell_measurement(self, state: torch.Tensor) -> Tuple[int, int]:
        """Bell measurement returning 2 classical bits."""
        probs = torch.abs(state.flatten()) ** 2
        probs = probs / probs.sum()
        outcome = np.random.choice(len(probs), p=probs.numpy())
        return (outcome // 2, outcome % 2)

    def _apply_pauli_correction(
        self, state: torch.Tensor, bits: Tuple[int, int]
    ) -> torch.Tensor:
        """Apply Pauli correction based on classical bits."""
        result = state.clone()
        if bits[1]:  # Pauli X
            result = torch.flip(result, [0])
        if bits[0]:  # Pauli Z
            result[1] *= -1
        return result

    def _destroy_bond(self, bond_id: bytes) -> None:
        """Destroy an entanglement bond."""
        if bond_id not in self.bonds:
            return

        bond = self.bonds[bond_id]

        for pattern in [bond.pattern_a, bond.pattern_b]:
            if pattern in self.pattern_bonds:
                self.pattern_bonds[pattern].discard(bond_id)

        del self.bonds[bond_id]

    def get_entangled_cluster(self, nh_id: bytes) -> Set[bytes]:
        """Get all patterns entangled (transitively) with a given pattern."""
        if nh_id not in self.pattern_bonds:
            return {nh_id}

        cluster = set()
        to_visit = {nh_id}

        while to_visit:
            current = to_visit.pop()
            if current in cluster:
                continue

            cluster.add(current)

            for bond_id in self.pattern_bonds.get(current, set()):
                bond = self.bonds.get(bond_id)
                if bond:
                    for pattern in [bond.pattern_a, bond.pattern_b]:
                        if pattern not in cluster:
                            to_visit.add(pattern)

        return cluster

    def entanglement_entropy(self, nh_id: bytes) -> float:
        """Calculate entanglement entropy (quantum connectivity)."""
        if nh_id not in self.pattern_bonds:
            return 0.0

        bond_ids = self.pattern_bonds[nh_id]
        if not bond_ids:
            return 0.0

        total_strength = sum(
            bond.strength * bond.fidelity()
            for bond_id in bond_ids
            if (bond := self.bonds.get(bond_id))
        )

        return np.log1p(total_strength)

    def propagate_gradient(
        self, gradient: torch.Tensor, source_pattern: bytes
    ) -> Dict[bytes, torch.Tensor]:
        """
        Propagate a gradient through entanglement bonds.

        Returns gradients for all entangled patterns.
        """
        propagated = {source_pattern: gradient}

        if source_pattern not in self.pattern_bonds:
            return propagated

        for bond_id in self.pattern_bonds[source_pattern]:
            bond = self.bonds.get(bond_id)
            if not bond:
                continue

            partner = (
                bond.pattern_b
                if bond.pattern_a == source_pattern
                else bond.pattern_a
            )

            # Attenuate by bond strength and fidelity
            propagated_grad = gradient * bond.strength * bond.fidelity()

            # Transform based on Bell type
            if bond.bell_type == BellState.PHI_MINUS:
                propagated_grad = -propagated_grad
            elif bond.bell_type == BellState.PSI_PLUS:
                propagated_grad = torch.flip(propagated_grad, [0])
            elif bond.bell_type == BellState.PSI_MINUS:
                propagated_grad = -torch.flip(propagated_grad, [0])

            propagated[partner] = propagated_grad

        return propagated

    def purify(self, bond_id: bytes, n_rounds: int = 3) -> EntanglementBond:
        """
        Purify an entanglement bond (improve fidelity).

        Args:
            bond_id: Bond to purify
            n_rounds: Number of purification rounds

        Returns:
            Purified bond
        """
        if bond_id not in self.bonds:
            raise ValueError(f"Unknown bond: {bond_id.hex()}")

        bond = self.bonds[bond_id]
        current_fidelity = bond.fidelity()

        for _ in range(n_rounds):
            if current_fidelity > 0.5:
                current_fidelity = current_fidelity**2 / (
                    current_fidelity**2 + (1 - current_fidelity) ** 2
                )

        bond.strength = current_fidelity
        return bond

    def get_bond(self, bond_id: bytes) -> Optional[EntanglementBond]:
        """Get a bond by ID."""
        return self.bonds.get(bond_id)

    def list_bonds(self, pattern_id: Optional[bytes] = None) -> List[EntanglementBond]:
        """List all bonds, optionally filtered by pattern."""
        if pattern_id:
            bond_ids = self.pattern_bonds.get(pattern_id, set())
            return [self.bonds[bid] for bid in bond_ids if bid in self.bonds]
        return list(self.bonds.values())

    def stats(self) -> Dict:
        """Get correlator statistics."""
        return {
            "total_bonds": len(self.bonds),
            "patterns_with_bonds": len(self.pattern_bonds),
            "teleportations": len(self.teleportation_log),
            "avg_fidelity": (
                np.mean([b.fidelity() for b in self.bonds.values()])
                if self.bonds
                else 0.0
            ),
        }
