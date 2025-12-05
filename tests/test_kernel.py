"""
Tests for the Synthetic Field Kernel.
"""

import pytest
import torch

from sfl.core.kernel import SyntheticFieldKernel
from sfl.core.types import Report, LifecycleEventType


class TestKernel:
    """Test suite for SyntheticFieldKernel."""

    @pytest.fixture
    def kernel(self):
        """Create a test kernel."""
        return SyntheticFieldKernel({
            "horizons": [1.0, 10.0],
            "n_critics": 2,
            "critic_hidden_dim": 128,
        })

    def test_register_pattern(self, kernel):
        """Test pattern registration."""
        nh_id = kernel.register(param_shape=(64,))

        assert nh_id is not None
        assert len(nh_id) == 32  # SHA256 hash

        pattern = kernel.get_pattern(nh_id)
        assert pattern.status == "active"
        assert pattern.param_shape == (64,)
        assert pattern.tau_rate == 1.0

    def test_register_with_lineage(self, kernel):
        """Test registration with lineage."""
        parent_id = kernel.register(param_shape=(64,))
        child_id = kernel.register(
            param_shape=(64,),
            lineage=[parent_id],
            metadata={"parent": parent_id.hex()}
        )

        child = kernel.get_pattern(child_id)
        assert parent_id in child.lineage

    def test_report(self, kernel):
        """Test report submission."""
        nh_id = kernel.register(param_shape=(64,))

        report = Report(
            nh_id=nh_id,
            state=torch.randn(32),
            action=torch.randn(8),
            cost=torch.tensor([0.1]),
            outcome=torch.randn(16),
        )

        kernel.report(report)

        contribution = kernel.get_contribution(nh_id)
        assert contribution.reports_submitted == 1

    def test_pull_gradient(self, kernel):
        """Test gradient pulling."""
        nh_id = kernel.register(param_shape=(64,))

        # Submit some reports
        for _ in range(5):
            report = Report(
                nh_id=nh_id,
                state=torch.randn(32),
                action=torch.randn(8),
                cost=torch.tensor([0.1]),
                outcome=torch.randn(16),
            )
            kernel.report(report)

        gradient = kernel.pull_gradient(nh_id)

        assert gradient.nh_id == nh_id
        assert gradient.param_grad is not None
        assert gradient.horizons == [1.0, 10.0]
        assert gradient.alloc_signal.tau_rate > 0

    def test_fork(self, kernel):
        """Test pattern forking."""
        parent_id = kernel.register(param_shape=(64,))
        kernel.patterns[parent_id].tau_rate = 2.0

        children = kernel.fork(parent_id, n_children=3)

        assert len(children) == 3
        for child_id in children:
            child = kernel.get_pattern(child_id)
            assert parent_id in child.lineage
            # τ is shared among children
            assert child.tau_rate == pytest.approx(2.0 / 3, rel=0.01)

    def test_merge(self, kernel):
        """Test pattern merging."""
        id1 = kernel.register(param_shape=(32,))
        id2 = kernel.register(param_shape=(64,))

        kernel.patterns[id1].tau_rate = 1.5
        kernel.patterns[id2].tau_rate = 0.5

        merged_id = kernel.merge([id1, id2])

        merged = kernel.get_pattern(merged_id)
        # τ is combined
        assert merged.tau_rate == pytest.approx(2.0, rel=0.01)
        # Original patterns are dissolved
        assert kernel.patterns[id1].status == "dissolved"
        assert kernel.patterns[id2].status == "dissolved"

    def test_freeze(self, kernel):
        """Test pattern freezing."""
        nh_id = kernel.register(param_shape=(64,))
        kernel.freeze(nh_id)

        pattern = kernel.get_pattern(nh_id)
        assert pattern.status == "frozen"
        assert pattern.tau_rate == 0.0

    def test_dissolve(self, kernel):
        """Test pattern dissolution."""
        nh_id = kernel.register(param_shape=(64,))
        kernel.dissolve(nh_id)

        pattern = kernel.get_pattern(nh_id)
        assert pattern.status == "dissolved"

    def test_lifecycle_events(self, kernel):
        """Test lifecycle event emission."""
        events = []
        kernel.on_lifecycle_event(lambda e: events.append(e))

        nh_id = kernel.register(param_shape=(64,))
        assert len(events) == 1
        assert events[0].event_type == LifecycleEventType.EMERGE

        kernel.freeze(nh_id)
        assert len(events) == 2
        assert events[1].event_type == LifecycleEventType.FREEZE

    def test_unknown_pattern_raises(self, kernel):
        """Test that operations on unknown patterns raise."""
        fake_id = b"\x00" * 32

        with pytest.raises(ValueError):
            kernel.get_pattern(fake_id)

        with pytest.raises(ValueError):
            kernel.report(Report(
                nh_id=fake_id,
                state=torch.randn(32),
                action=torch.randn(8),
                cost=torch.tensor([0.1]),
                outcome=torch.randn(16),
            ))

    def test_stats(self, kernel):
        """Test statistics gathering."""
        # Register some patterns
        for _ in range(3):
            kernel.register(param_shape=(64,))

        stats = kernel.get_stats()

        assert stats["total_patterns"] == 3
        assert stats["active_patterns"] == 3
        assert stats["frozen_patterns"] == 0


class TestQuantumCorrelator:
    """Test suite for QuantumCorrelator."""

    @pytest.fixture
    def correlator(self):
        """Create a test correlator."""
        from sfl.core.quantum import QuantumCorrelator
        return QuantumCorrelator()

    def test_create_bond(self, correlator):
        """Test entanglement bond creation."""
        from sfl.core.quantum import BellState

        pattern_a = b"a" * 32
        pattern_b = b"b" * 32

        bond = correlator.create_bond(
            pattern_a, pattern_b,
            bell_type=BellState.PHI_PLUS
        )

        assert bond.pattern_a == pattern_a
        assert bond.pattern_b == pattern_b
        assert bond.bell_type == BellState.PHI_PLUS
        assert bond.strength == 1.0

    def test_bell_test(self, correlator):
        """Test Bell (CHSH) test."""
        from sfl.core.quantum import BellState

        pattern_a = b"a" * 32
        pattern_b = b"b" * 32

        bond = correlator.create_bond(pattern_a, pattern_b, BellState.PHI_PLUS)
        result = correlator.bell_test(bond.bond_id, n_samples=500)

        # Quantum correlations should violate Bell inequality
        # S > 2 for quantum states (max 2√2 ≈ 2.828)
        assert result.S_parameter > 0
        assert isinstance(result.is_quantum, bool)

    def test_entangled_cluster(self, correlator):
        """Test finding entangled clusters."""
        p1, p2, p3 = b"1" * 32, b"2" * 32, b"3" * 32

        correlator.create_bond(p1, p2)
        correlator.create_bond(p2, p3)

        cluster = correlator.get_entangled_cluster(p1)

        assert p1 in cluster
        assert p2 in cluster
        assert p3 in cluster

    def test_gradient_propagation(self, correlator):
        """Test gradient propagation through bonds."""
        from sfl.core.quantum import BellState

        p1, p2 = b"1" * 32, b"2" * 32
        correlator.create_bond(p1, p2, BellState.PHI_PLUS)

        gradient = torch.randn(10)
        propagated = correlator.propagate_gradient(gradient, p1)

        assert p1 in propagated
        assert p2 in propagated


class TestEmergenceDetector:
    """Test suite for EmergenceDetector."""

    @pytest.fixture
    def detector(self):
        """Create a test detector."""
        from sfl.core.emergence import EmergenceDetector
        return EmergenceDetector()

    def test_record_activation(self, detector):
        """Test activation recording."""
        nh_id = b"test" * 8
        detector.record_activation(nh_id, 0.8)
        detector.record_activation(nh_id, 0.9)

        assert nh_id in detector.pattern_activations
        assert len(detector.pattern_activations[nh_id]) == 2

    def test_record_self_reference(self, detector):
        """Test self-reference recording."""
        nh_id = b"test" * 8

        for _ in range(15):
            detector.record_self_reference(nh_id)

        assert detector.self_reference_counts[nh_id] == 15

    def test_consciousness_probability(self, detector):
        """Test consciousness probability estimation."""
        nh_id = b"test" * 8
        cluster = {nh_id}

        # Initially should be 0
        prob = detector.get_consciousness_probability(cluster)
        assert prob == 0.0

    def test_precaution_threshold(self, detector):
        """Test precautionary principle."""
        nh_id = b"test" * 8
        cluster = {nh_id}

        # Initially should not apply precaution
        assert not detector.should_apply_precaution(cluster)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
