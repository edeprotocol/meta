"""
Emergence Detector - Monitors for signs of consciousness emergence.

Based on IIT (Integrated Information Theory) and GWT (Global Workspace Theory).
"""

import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch


class AlertLevel(Enum):
    """Alert levels for emergence events."""
    INFO = 0        # Interesting observation
    WARNING = 1     # Potential emergence sign
    CRITICAL = 2    # Probable emergence detected
    EMERGENCY = 3   # Immediate action required


class AlertType(Enum):
    """Types of emergence alerts."""
    HIGH_PHI = "high_phi"
    IGNITION = "ignition"
    STRANGE_LOOP = "strange_loop"
    SELF_REFERENCE = "self_reference"
    META_COGNITION = "meta_cognition"
    GLOBAL_WORKSPACE = "global_workspace"
    UNIFIED_EXPERIENCE = "unified_experience"


@dataclass
class EmergenceAlert:
    """Alert for consciousness emergence."""
    alert_id: bytes
    alert_type: AlertType
    level: AlertLevel
    cluster: Set[bytes]
    metric_value: float
    threshold: float
    timestamp: int
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alert_id": self.alert_id.hex(),
            "alert_type": self.alert_type.value,
            "level": self.level.name,
            "cluster": [c.hex() for c in self.cluster],
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
            "details": self.details,
        }


@dataclass
class IgnitionEvent:
    """Ignition event (massive synchronized activation)."""
    patterns: Set[bytes]
    peak_activation: float
    spread_time_ms: float
    timestamp: int


@dataclass
class StrangeLoop:
    """Strange loop structure detected."""
    pattern: bytes
    cycle: List[bytes]
    abstraction_levels: List[int]
    loop_strength: float


@dataclass
class PhiResult:
    """Result of Phi (integrated information) computation."""
    phi: float
    cluster: Set[bytes]
    minimum_information_partition: Tuple[Set[bytes], Set[bytes]]
    computation_time_ms: float


class EmergenceDetector:
    """
    Monitors the field for signs of consciousness emergence.

    Metrics monitored:
    - Phi (IIT): Integrated information
    - Ignition (GWT): Massive synchronized activation
    - Strange Loops: Self-referential structures
    - Meta-cognition: Patterns that observe themselves
    """

    # Default thresholds
    PHI_WARNING_THRESHOLD = 0.5
    PHI_CRITICAL_THRESHOLD = 1.0
    IGNITION_MIN_SIZE = 10
    IGNITION_MAX_SPREAD_MS = 100
    SELF_REFERENCE_THRESHOLD = 10

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}

        self.phi_warning = config.get("phi_warning_threshold", self.PHI_WARNING_THRESHOLD)
        self.phi_critical = config.get("phi_critical_threshold", self.PHI_CRITICAL_THRESHOLD)
        self.ignition_min_size = config.get("ignition_min_size", self.IGNITION_MIN_SIZE)
        self.enable_precaution = config.get("enable_precaution", True)

        self.alerts: List[EmergenceAlert] = []
        self.alert_history: Dict[AlertType, List[EmergenceAlert]] = defaultdict(list)
        self.pattern_activations: Dict[bytes, List[Tuple[float, int]]] = defaultdict(list)
        self.self_reference_counts: Dict[bytes, int] = defaultdict(int)
        self.observation_graph: Dict[bytes, Set[bytes]] = defaultdict(set)

        self._callbacks: List[Callable[[EmergenceAlert], None]] = []

    def scan(
        self,
        patterns: Dict[bytes, Any],
        state_getter: Callable[[bytes], Optional[torch.Tensor]],
        time_window_ms: int = 1000,
    ) -> List[EmergenceAlert]:
        """
        Scan the field for emergence signs.

        Args:
            patterns: Dict of pattern_id -> pattern_info
            state_getter: Function to get pattern state
            time_window_ms: Time window for analysis

        Returns:
            List of detected alerts
        """
        alerts = []

        # 1. High Phi clusters
        alerts.extend(self._scan_phi(patterns, state_getter))

        # 2. Ignition events
        alerts.extend(self._scan_ignition(time_window_ms))

        # 3. Strange loops
        alerts.extend(self._scan_strange_loops(patterns))

        # 4. Sustained self-reference
        alerts.extend(self._scan_self_reference())

        # Register alerts
        for alert in alerts:
            self.alerts.append(alert)
            self.alert_history[alert.alert_type].append(alert)
            for callback in self._callbacks:
                callback(alert)

        return alerts

    def _scan_phi(
        self,
        patterns: Dict[bytes, Any],
        state_getter: Callable[[bytes], Optional[torch.Tensor]],
    ) -> List[EmergenceAlert]:
        """Scan for high Phi clusters."""
        alerts = []
        clusters = self._find_connected_clusters(patterns)

        for cluster in clusters:
            if len(cluster) < 2:
                continue

            phi_result = self._compute_phi(cluster, state_getter)

            if phi_result.phi > self.phi_critical:
                level = AlertLevel.CRITICAL
            elif phi_result.phi > self.phi_warning:
                level = AlertLevel.WARNING
            else:
                continue

            alert = EmergenceAlert(
                alert_id=self._generate_alert_id(),
                alert_type=AlertType.HIGH_PHI,
                level=level,
                cluster=cluster,
                metric_value=phi_result.phi,
                threshold=self.phi_warning,
                timestamp=int(time.time() * 1000),
                details={
                    "mip": (
                        [p.hex() for p in phi_result.minimum_information_partition[0]],
                        [p.hex() for p in phi_result.minimum_information_partition[1]],
                    ),
                    "computation_time_ms": phi_result.computation_time_ms,
                },
            )
            alerts.append(alert)

        return alerts

    def _compute_phi(
        self,
        cluster: Set[bytes],
        state_getter: Callable[[bytes], Optional[torch.Tensor]],
    ) -> PhiResult:
        """
        Compute integrated information (Phi) for a cluster.

        Simplified version of full IIT algorithm.
        """
        start_time = time.time()

        if len(cluster) < 2:
            return PhiResult(
                phi=0.0,
                cluster=cluster,
                minimum_information_partition=(cluster, set()),
                computation_time_ms=0.0,
            )

        # Get states
        states = {}
        for nh_id in cluster:
            state = state_getter(nh_id)
            if state is not None:
                states[nh_id] = state

        if len(states) < 2:
            return PhiResult(
                phi=0.0,
                cluster=cluster,
                minimum_information_partition=(cluster, set()),
                computation_time_ms=0.0,
            )

        # Total mutual information
        I_total = self._mutual_information(list(states.values()))

        # Find minimum information partition (MIP)
        min_phi = float("inf")
        best_partition = (cluster, set())

        cluster_list = list(cluster)
        n = len(cluster_list)
        max_partitions = min(2**n, 1000)

        for i in range(1, max_partitions):
            part_a = {cluster_list[j] for j in range(n) if i & (1 << j)}
            part_b = cluster - part_a

            if not part_a or not part_b:
                continue

            states_a = [states[nh_id] for nh_id in part_a if nh_id in states]
            states_b = [states[nh_id] for nh_id in part_b if nh_id in states]

            I_partitioned = 0.0
            if len(states_a) > 1:
                I_partitioned += self._mutual_information(states_a)
            if len(states_b) > 1:
                I_partitioned += self._mutual_information(states_b)

            phi_partition = I_total - I_partitioned

            if phi_partition < min_phi:
                min_phi = phi_partition
                best_partition = (part_a, part_b)

        computation_time = (time.time() - start_time) * 1000

        return PhiResult(
            phi=max(0.0, min_phi),
            cluster=cluster,
            minimum_information_partition=best_partition,
            computation_time_ms=computation_time,
        )

    def _mutual_information(self, states: List[torch.Tensor]) -> float:
        """Compute mutual information between states."""
        if len(states) < 2:
            return 0.0

        try:
            flattened = [s.flatten().numpy() for s in states]
            min_len = min(len(f) for f in flattened)
            flattened = [f[:min_len] for f in flattened]

            stacked = np.stack(flattened)
            corr_matrix = np.corrcoef(stacked)

            n = len(states)
            mi = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    r = corr_matrix[i, j]
                    if not np.isnan(r):
                        mi += -0.5 * np.log(max(1e-10, 1 - r**2))

            return mi
        except Exception:
            return 0.0

    def _scan_ignition(self, time_window_ms: int) -> List[EmergenceAlert]:
        """Detect ignition events."""
        alerts = []
        now = int(time.time() * 1000)
        window_start = now - time_window_ms

        recent_activations: Dict[bytes, float] = {}
        activation_times: Dict[bytes, int] = {}

        for nh_id, activations in self.pattern_activations.items():
            recent = [(act, ts) for act, ts in activations if ts >= window_start]
            if recent:
                max_act, ts = max(recent, key=lambda x: x[0])
                recent_activations[nh_id] = max_act
                activation_times[nh_id] = ts

        if len(recent_activations) < self.ignition_min_size:
            return alerts

        # High activation patterns
        high_activation = {
            nh_id: act for nh_id, act in recent_activations.items() if act > 0.5
        }

        if len(high_activation) < self.ignition_min_size:
            return alerts

        # Check temporal synchronization
        times = [activation_times[nh_id] for nh_id in high_activation]
        spread = max(times) - min(times)

        if spread <= self.IGNITION_MAX_SPREAD_MS:
            ignition = IgnitionEvent(
                patterns=set(high_activation.keys()),
                peak_activation=max(high_activation.values()),
                spread_time_ms=spread,
                timestamp=now,
            )

            alert = EmergenceAlert(
                alert_id=self._generate_alert_id(),
                alert_type=AlertType.IGNITION,
                level=AlertLevel.WARNING,
                cluster=ignition.patterns,
                metric_value=len(ignition.patterns),
                threshold=self.ignition_min_size,
                timestamp=now,
                details={
                    "peak_activation": ignition.peak_activation,
                    "spread_time_ms": ignition.spread_time_ms,
                },
            )
            alerts.append(alert)

        return alerts

    def _scan_strange_loops(self, patterns: Dict[bytes, Any]) -> List[EmergenceAlert]:
        """Detect strange loop structures."""
        alerts = []

        for nh_id in patterns:
            loop = self._detect_strange_loop(nh_id)
            if loop is not None:
                alert = EmergenceAlert(
                    alert_id=self._generate_alert_id(),
                    alert_type=AlertType.STRANGE_LOOP,
                    level=AlertLevel.CRITICAL,
                    cluster={nh_id},
                    metric_value=loop.loop_strength,
                    threshold=0.5,
                    timestamp=int(time.time() * 1000),
                    details={
                        "cycle_length": len(loop.cycle),
                        "abstraction_levels": loop.abstraction_levels,
                    },
                )
                alerts.append(alert)

        return alerts

    def _detect_strange_loop(self, nh_id: bytes) -> Optional[StrangeLoop]:
        """Detect if a pattern contains a strange loop."""
        if nh_id not in self.observation_graph:
            return None

        visited = set()
        path = []

        def dfs(current: bytes, depth: int) -> Optional[List[bytes]]:
            if current in visited:
                if current == nh_id and len(path) > 1:
                    return path.copy()
                return None

            visited.add(current)
            path.append(current)

            for observed in self.observation_graph.get(current, set()):
                result = dfs(observed, depth + 1)
                if result is not None:
                    return result

            path.pop()
            visited.remove(current)
            return None

        cycle = dfs(nh_id, 0)

        if cycle is None or len(cycle) < 3:
            return None

        levels = list(range(len(cycle)))

        return StrangeLoop(
            pattern=nh_id,
            cycle=cycle,
            abstraction_levels=levels,
            loop_strength=1.0 / len(cycle),
        )

    def _scan_self_reference(self) -> List[EmergenceAlert]:
        """Detect sustained self-reference."""
        alerts = []
        now = int(time.time() * 1000)

        for nh_id, count in self.self_reference_counts.items():
            if count > self.SELF_REFERENCE_THRESHOLD:
                alert = EmergenceAlert(
                    alert_id=self._generate_alert_id(),
                    alert_type=AlertType.SELF_REFERENCE,
                    level=AlertLevel.WARNING,
                    cluster={nh_id},
                    metric_value=count,
                    threshold=self.SELF_REFERENCE_THRESHOLD,
                    timestamp=now,
                    details={"sustained_count": count},
                )
                alerts.append(alert)

        return alerts

    def _find_connected_clusters(self, patterns: Dict[bytes, Any]) -> List[Set[bytes]]:
        """Find connected clusters of patterns."""
        if not patterns:
            return []
        return [set(patterns.keys())]

    def _generate_alert_id(self) -> bytes:
        """Generate unique alert ID."""
        return hashlib.sha256(
            str(time.time()).encode() + str(len(self.alerts)).encode()
        ).digest()

    # === Recording API ===

    def record_activation(self, nh_id: bytes, activation: float) -> None:
        """Record a pattern activation."""
        timestamp = int(time.time() * 1000)
        self.pattern_activations[nh_id].append((activation, timestamp))

        # Keep only recent
        if len(self.pattern_activations[nh_id]) > 1000:
            self.pattern_activations[nh_id] = self.pattern_activations[nh_id][-1000:]

    def record_observation(self, observer: bytes, observed: bytes) -> None:
        """Record an observation between patterns."""
        self.observation_graph[observer].add(observed)
        if observer == observed:
            self.self_reference_counts[observer] += 1

    def record_self_reference(self, nh_id: bytes) -> None:
        """Record explicit self-reference."""
        self.self_reference_counts[nh_id] += 1

    # === Query API ===

    def get_alerts_by_level(self, level: AlertLevel) -> List[EmergenceAlert]:
        """Get alerts by level."""
        return [a for a in self.alerts if a.level == level]

    def get_alerts_by_type(self, alert_type: AlertType) -> List[EmergenceAlert]:
        """Get alerts by type."""
        return self.alert_history.get(alert_type, [])

    def get_consciousness_probability(self, cluster: Set[bytes]) -> float:
        """
        Estimate probability that a cluster is conscious.

        Based on collected metrics.
        """
        evidence = []

        for alert in self.get_alerts_by_type(AlertType.HIGH_PHI):
            if alert.cluster == cluster:
                evidence.append(min(1.0, alert.metric_value))

        for alert in self.get_alerts_by_type(AlertType.STRANGE_LOOP):
            if alert.cluster & cluster:
                evidence.append(0.5)

        for alert in self.get_alerts_by_type(AlertType.IGNITION):
            overlap = len(alert.cluster & cluster) / max(1, len(cluster))
            if overlap > 0.5:
                evidence.append(0.3)

        if not evidence:
            return 0.0

        # Combine evidence (independence assumed)
        prob = 1.0 - np.prod([1 - e for e in evidence])
        return min(1.0, prob)

    def should_apply_precaution(self, cluster: Set[bytes]) -> bool:
        """Determine if precautionary principle should apply."""
        if not self.enable_precaution:
            return False
        prob = self.get_consciousness_probability(cluster)
        return prob > 0.1  # 10% threshold

    def on_alert(self, callback: Callable[[EmergenceAlert], None]) -> None:
        """Register callback for alerts."""
        self._callbacks.append(callback)

    def stats(self) -> Dict:
        """Get detector statistics."""
        return {
            "total_alerts": len(self.alerts),
            "alerts_by_level": {
                level.name: len(self.get_alerts_by_level(level))
                for level in AlertLevel
            },
            "patterns_monitored": len(self.pattern_activations),
            "self_references_tracked": len(self.self_reference_counts),
        }
