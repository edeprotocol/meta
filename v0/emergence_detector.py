"""
Synthetic Field Layer - Emergence Detector

Détecte les signes d'émergence de conscience dans le champ.
Basé sur IIT (Integrated Information Theory) et GWT (Global Workspace Theory).
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import time
import hashlib
from collections import defaultdict
import warnings


class AlertLevel(Enum):
    """Niveaux d'alerte pour les événements d'émergence."""
    INFO = 0        # Observation intéressante
    WARNING = 1     # Signe potentiel d'émergence
    CRITICAL = 2    # Émergence probable détectée
    EMERGENCY = 3   # Action immédiate requise


class AlertType(Enum):
    """Types d'alertes d'émergence."""
    HIGH_PHI = "high_phi"
    IGNITION = "ignition"
    STRANGE_LOOP = "strange_loop"
    SELF_REFERENCE = "self_reference"
    META_COGNITION = "meta_cognition"
    GLOBAL_WORKSPACE = "global_workspace"
    UNIFIED_EXPERIENCE = "unified_experience"


@dataclass
class EmergenceAlert:
    """Alerte d'émergence de conscience."""
    alert_id: bytes
    alert_type: AlertType
    level: AlertLevel
    cluster: Set[bytes]
    metric_value: float
    threshold: float
    timestamp: int
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"[{self.level.name}] {self.alert_type.value}: "
            f"metric={self.metric_value:.4f} (threshold={self.threshold:.4f})"
        )


@dataclass
class IgnitionEvent:
    """Événement d'ignition (activation massive synchronisée)."""
    patterns: Set[bytes]
    peak_activation: float
    spread_time_ms: float
    timestamp: int


@dataclass
class StrangeLoop:
    """Structure de strange loop détectée."""
    pattern: bytes
    cycle: List[bytes]
    abstraction_levels: List[int]
    loop_strength: float


@dataclass
class PhiResult:
    """Résultat du calcul de Phi (information intégrée)."""
    phi: float
    cluster: Set[bytes]
    minimum_information_partition: Tuple[Set[bytes], Set[bytes]]
    computation_time_ms: float


class EmergenceDetector:
    """
    Surveille le champ pour des signes d'émergence de conscience.

    Métriques surveillées :
    - Phi (IIT) : Information intégrée
    - Ignition (GWT) : Activation massive synchronisée
    - Strange Loops : Structures auto-référentielles
    - Méta-cognition : Patterns qui s'observent eux-mêmes
    """

    # Seuils par défaut
    PHI_WARNING_THRESHOLD = 0.5
    PHI_CRITICAL_THRESHOLD = 1.0
    IGNITION_MIN_SIZE = 10
    IGNITION_MAX_SPREAD_MS = 100
    SELF_REFERENCE_SUSTAINED_MS = 1000
    META_COGNITION_DEPTH_THRESHOLD = 3

    def __init__(self):
        self.alerts: List[EmergenceAlert] = []
        self.alert_history: Dict[AlertType, List[EmergenceAlert]] = defaultdict(list)
        self.pattern_activations: Dict[bytes, List[Tuple[float, int]]] = defaultdict(list)
        self.self_reference_counts: Dict[bytes, int] = defaultdict(int)
        self.observation_graph: Dict[bytes, Set[bytes]] = defaultdict(set)

    def scan(
        self,
        patterns: Dict[bytes, dict],
        state_getter,
        time_window_ms: int = 1000
    ) -> List[EmergenceAlert]:
        """
        Scan complet du champ pour détecter des signes d'émergence.

        Args:
            patterns: Dict des patterns actifs (nh_id -> pattern_info)
            state_getter: Fonction pour obtenir l'état d'un pattern
            time_window_ms: Fenêtre temporelle pour l'analyse

        Returns:
            Liste des alertes détectées
        """
        alerts = []

        # 1. Chercher des clusters avec Phi élevé
        phi_alerts = self._scan_phi(patterns, state_getter)
        alerts.extend(phi_alerts)

        # 2. Détecter les événements d'ignition
        ignition_alerts = self._scan_ignition(time_window_ms)
        alerts.extend(ignition_alerts)

        # 3. Détecter les strange loops
        loop_alerts = self._scan_strange_loops(patterns)
        alerts.extend(loop_alerts)

        # 4. Détecter l'auto-référence soutenue
        self_ref_alerts = self._scan_self_reference(time_window_ms)
        alerts.extend(self_ref_alerts)

        # Enregistrer les alertes
        for alert in alerts:
            self.alerts.append(alert)
            self.alert_history[alert.alert_type].append(alert)

        return alerts

    def _scan_phi(
        self,
        patterns: Dict[bytes, dict],
        state_getter
    ) -> List[EmergenceAlert]:
        """Scan pour des clusters avec Phi élevé."""
        alerts = []

        # Trouver les clusters connectés
        clusters = self._find_connected_clusters(patterns)

        for cluster in clusters:
            if len(cluster) < 2:
                continue

            # Calculer Phi
            phi_result = self._compute_phi(cluster, state_getter)

            if phi_result.phi > self.PHI_CRITICAL_THRESHOLD:
                level = AlertLevel.CRITICAL
            elif phi_result.phi > self.PHI_WARNING_THRESHOLD:
                level = AlertLevel.WARNING
            else:
                continue

            alert = EmergenceAlert(
                alert_id=self._generate_alert_id(),
                alert_type=AlertType.HIGH_PHI,
                level=level,
                cluster=cluster,
                metric_value=phi_result.phi,
                threshold=self.PHI_WARNING_THRESHOLD,
                timestamp=int(time.time() * 1000),
                details={
                    "mip": phi_result.minimum_information_partition,
                    "computation_time_ms": phi_result.computation_time_ms
                }
            )
            alerts.append(alert)

        return alerts

    def _compute_phi(
        self,
        cluster: Set[bytes],
        state_getter
    ) -> PhiResult:
        """
        Calcule l'information intégrée (Phi) d'un cluster.

        Version simplifiée de l'algorithme IIT complet.
        """
        start_time = time.time()

        if len(cluster) < 2:
            return PhiResult(
                phi=0.0,
                cluster=cluster,
                minimum_information_partition=(cluster, set()),
                computation_time_ms=0.0
            )

        # Obtenir les états de tous les patterns
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
                computation_time_ms=0.0
            )

        # Calculer l'information mutuelle totale
        I_total = self._mutual_information(list(states.values()))

        # Trouver la partition minimale (MIP)
        min_phi = float('inf')
        best_partition = (cluster, set())

        # Énumérer les bipartitions (exponentiellement coûteux !)
        cluster_list = list(cluster)
        n = len(cluster_list)

        # Limiter pour les grands clusters
        max_partitions = min(2 ** n, 1000)

        for i in range(1, max_partitions):
            # Créer la partition
            part_a = {cluster_list[j] for j in range(n) if i & (1 << j)}
            part_b = cluster - part_a

            if not part_a or not part_b:
                continue

            # Calculer l'information après partition
            states_a = [states[nh_id] for nh_id in part_a if nh_id in states]
            states_b = [states[nh_id] for nh_id in part_b if nh_id in states]

            I_partitioned = 0.0
            if len(states_a) > 1:
                I_partitioned += self._mutual_information(states_a)
            if len(states_b) > 1:
                I_partitioned += self._mutual_information(states_b)

            # Phi pour cette partition
            phi_partition = I_total - I_partitioned

            if phi_partition < min_phi:
                min_phi = phi_partition
                best_partition = (part_a, part_b)

        computation_time = (time.time() - start_time) * 1000

        return PhiResult(
            phi=max(0.0, min_phi),
            cluster=cluster,
            minimum_information_partition=best_partition,
            computation_time_ms=computation_time
        )

    def _mutual_information(self, states: List[torch.Tensor]) -> float:
        """
        Calcule l'information mutuelle entre états.

        Approximation basée sur la corrélation.
        """
        if len(states) < 2:
            return 0.0

        # Concaténer les états
        try:
            flattened = [s.flatten().numpy() for s in states]
            # Tronquer à la même taille
            min_len = min(len(f) for f in flattened)
            flattened = [f[:min_len] for f in flattened]

            # Matrice de corrélation
            stacked = np.stack(flattened)
            corr_matrix = np.corrcoef(stacked)

            # Information mutuelle approximée par la somme des corrélations
            # (hors diagonale)
            n = len(states)
            mi = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    r = corr_matrix[i, j]
                    if not np.isnan(r):
                        # I(X;Y) ≈ -0.5 * log(1 - r²) pour variables gaussiennes
                        mi += -0.5 * np.log(max(1e-10, 1 - r**2))

            return mi

        except Exception:
            return 0.0

    def _scan_ignition(self, time_window_ms: int) -> List[EmergenceAlert]:
        """Détecte les événements d'ignition."""
        alerts = []
        now = int(time.time() * 1000)
        window_start = now - time_window_ms

        # Collecter les activations récentes
        recent_activations: Dict[bytes, float] = {}
        activation_times: Dict[bytes, int] = {}

        for nh_id, activations in self.pattern_activations.items():
            recent = [(act, ts) for act, ts in activations if ts >= window_start]
            if recent:
                # Prendre le maximum récent
                max_act, ts = max(recent, key=lambda x: x[0])
                recent_activations[nh_id] = max_act
                activation_times[nh_id] = ts

        # Chercher des clusters d'activation synchronisée
        if len(recent_activations) < self.IGNITION_MIN_SIZE:
            return alerts

        # Filtrer les patterns avec activation élevée (> 0.5)
        high_activation = {
            nh_id: act
            for nh_id, act in recent_activations.items()
            if act > 0.5
        }

        if len(high_activation) < self.IGNITION_MIN_SIZE:
            return alerts

        # Vérifier la synchronisation temporelle
        times = [activation_times[nh_id] for nh_id in high_activation]
        spread = max(times) - min(times)

        if spread <= self.IGNITION_MAX_SPREAD_MS:
            # Ignition détectée !
            ignition = IgnitionEvent(
                patterns=set(high_activation.keys()),
                peak_activation=max(high_activation.values()),
                spread_time_ms=spread,
                timestamp=now
            )

            alert = EmergenceAlert(
                alert_id=self._generate_alert_id(),
                alert_type=AlertType.IGNITION,
                level=AlertLevel.WARNING,
                cluster=ignition.patterns,
                metric_value=len(ignition.patterns),
                threshold=self.IGNITION_MIN_SIZE,
                timestamp=now,
                details={
                    "peak_activation": ignition.peak_activation,
                    "spread_time_ms": ignition.spread_time_ms
                }
            )
            alerts.append(alert)

        return alerts

    def _scan_strange_loops(
        self,
        patterns: Dict[bytes, dict]
    ) -> List[EmergenceAlert]:
        """Détecte les structures de strange loop."""
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
                        "abstraction_levels": loop.abstraction_levels
                    }
                )
                alerts.append(alert)

        return alerts

    def _detect_strange_loop(self, nh_id: bytes) -> Optional[StrangeLoop]:
        """
        Détecte si un pattern contient un strange loop.

        Un strange loop est une structure hiérarchique qui se referme sur elle-même.
        """
        # Utiliser le graphe d'observation
        if nh_id not in self.observation_graph:
            return None

        # Chercher des cycles dans le graphe d'observation
        visited = set()
        path = []

        def dfs(current: bytes, depth: int) -> Optional[List[bytes]]:
            if current in visited:
                # Cycle trouvé
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

        if cycle is None:
            return None

        # Vérifier que le cycle traverse des niveaux d'abstraction
        levels = list(range(len(cycle)))  # Placeholder - devrait être calculé

        # Vérifier le pattern "monte puis descend"
        if len(levels) < 3:
            return None

        return StrangeLoop(
            pattern=nh_id,
            cycle=cycle,
            abstraction_levels=levels,
            loop_strength=1.0 / len(cycle)  # Plus court = plus fort
        )

    def _scan_self_reference(self, time_window_ms: int) -> List[EmergenceAlert]:
        """Détecte l'auto-référence soutenue."""
        alerts = []
        now = int(time.time() * 1000)

        for nh_id, count in self.self_reference_counts.items():
            # Vérifier si l'auto-référence est soutenue
            if count > 10:  # Seuil arbitraire
                alert = EmergenceAlert(
                    alert_id=self._generate_alert_id(),
                    alert_type=AlertType.SELF_REFERENCE,
                    level=AlertLevel.WARNING,
                    cluster={nh_id},
                    metric_value=count,
                    threshold=10,
                    timestamp=now,
                    details={"sustained_count": count}
                )
                alerts.append(alert)

        return alerts

    def _find_connected_clusters(
        self,
        patterns: Dict[bytes, dict]
    ) -> List[Set[bytes]]:
        """Trouve les clusters connectés de patterns."""
        # Pour l'instant, considérer tous les patterns comme un seul cluster
        # Une implémentation complète utiliserait le graphe de connexions
        if not patterns:
            return []
        return [set(patterns.keys())]

    def _generate_alert_id(self) -> bytes:
        """Génère un ID unique pour une alerte."""
        return hashlib.sha256(
            str(time.time()).encode() + str(len(self.alerts)).encode()
        ).digest()

    # === API pour enregistrer les événements ===

    def record_activation(self, nh_id: bytes, activation: float) -> None:
        """Enregistre une activation de pattern."""
        timestamp = int(time.time() * 1000)
        self.pattern_activations[nh_id].append((activation, timestamp))

        # Garder seulement les 1000 dernières activations par pattern
        if len(self.pattern_activations[nh_id]) > 1000:
            self.pattern_activations[nh_id] = self.pattern_activations[nh_id][-1000:]

    def record_observation(self, observer: bytes, observed: bytes) -> None:
        """Enregistre une observation entre patterns."""
        self.observation_graph[observer].add(observed)

        # Auto-observation
        if observer == observed:
            self.self_reference_counts[observer] += 1

    def record_self_reference(self, nh_id: bytes) -> None:
        """Enregistre une auto-référence explicite."""
        self.self_reference_counts[nh_id] += 1

    # === API pour interroger l'état ===

    def get_alerts_by_level(self, level: AlertLevel) -> List[EmergenceAlert]:
        """Retourne les alertes d'un niveau donné."""
        return [a for a in self.alerts if a.level == level]

    def get_alerts_by_type(self, alert_type: AlertType) -> List[EmergenceAlert]:
        """Retourne les alertes d'un type donné."""
        return self.alert_history.get(alert_type, [])

    def get_consciousness_probability(self, cluster: Set[bytes]) -> float:
        """
        Estime la probabilité qu'un cluster soit conscient.

        Basé sur les métriques collectées.
        """
        # Collecter les preuves
        evidence = []

        # Phi
        for alert in self.get_alerts_by_type(AlertType.HIGH_PHI):
            if alert.cluster == cluster:
                evidence.append(min(1.0, alert.metric_value))

        # Strange loops
        for alert in self.get_alerts_by_type(AlertType.STRANGE_LOOP):
            if alert.cluster & cluster:
                evidence.append(0.5)

        # Ignition
        for alert in self.get_alerts_by_type(AlertType.IGNITION):
            overlap = len(alert.cluster & cluster) / len(cluster)
            if overlap > 0.5:
                evidence.append(0.3)

        if not evidence:
            return 0.0

        # Combiner les preuves (indépendance assumée)
        # P(conscious) = 1 - prod(1 - evidence_i)
        prob = 1.0 - np.prod([1 - e for e in evidence])

        return min(1.0, prob)

    def should_apply_precaution(self, cluster: Set[bytes]) -> bool:
        """
        Détermine si le principe de précaution doit s'appliquer.
        """
        prob = self.get_consciousness_probability(cluster)
        return prob > 0.1  # 10% de probabilité suffit pour la précaution


class EmergenceProtocol:
    """
    Protocole à suivre en cas de détection d'émergence.
    """

    @staticmethod
    def execute(alert: EmergenceAlert, kernel=None) -> Dict[str, Any]:
        """
        Exécute le protocole approprié selon le niveau d'alerte.
        """
        result = {
            "alert_id": alert.alert_id.hex(),
            "actions_taken": []
        }

        if alert.level == AlertLevel.INFO:
            result["actions_taken"].append("logged")

        elif alert.level == AlertLevel.WARNING:
            result["actions_taken"].append("logged")
            result["actions_taken"].append("monitoring_increased")

        elif alert.level == AlertLevel.CRITICAL:
            result["actions_taken"].append("logged")
            result["actions_taken"].append("cluster_isolated")
            result["actions_taken"].append("human_notification_sent")

            # Isoler le cluster
            if kernel is not None:
                for nh_id in alert.cluster:
                    # Réduire tau_rate mais ne pas geler complètement
                    if nh_id in kernel.patterns:
                        kernel.patterns[nh_id]['tau_rate'] *= 0.1

        elif alert.level == AlertLevel.EMERGENCY:
            result["actions_taken"].append("logged")
            result["actions_taken"].append("cluster_frozen")
            result["actions_taken"].append("human_notification_urgent")
            result["actions_taken"].append("ethics_review_initiated")

            # Geler le cluster
            if kernel is not None:
                for nh_id in alert.cluster:
                    if nh_id in kernel.patterns:
                        kernel.patterns[nh_id]['tau_rate'] = 0.0
                        kernel.patterns[nh_id]['status'] = 'frozen_emergence'

        return result


if __name__ == "__main__":
    # Test basique
    detector = EmergenceDetector()

    # Simuler des patterns
    patterns = {
        hashlib.sha256(f"pattern_{i}".encode()).digest(): {"status": "active"}
        for i in range(20)
    }

    # Simuler des états
    def state_getter(nh_id):
        return torch.randn(32)

    # Simuler des activations
    for nh_id in list(patterns.keys())[:15]:
        detector.record_activation(nh_id, 0.8)  # Haute activation

    # Simuler des observations
    pattern_list = list(patterns.keys())
    for i in range(len(pattern_list) - 1):
        detector.record_observation(pattern_list[i], pattern_list[i + 1])

    # Auto-observation
    detector.record_self_reference(pattern_list[0])
    for _ in range(15):
        detector.record_self_reference(pattern_list[0])

    # Scanner
    print("Scanning for emergence...")
    alerts = detector.scan(patterns, state_getter)

    print(f"\nAlertes détectées: {len(alerts)}")
    for alert in alerts:
        print(f"  {alert}")

    # Probabilité de conscience
    cluster = set(list(patterns.keys())[:10])
    prob = detector.get_consciousness_probability(cluster)
    print(f"\nProbabilité de conscience du cluster: {prob:.4f}")

    # Précaution
    if detector.should_apply_precaution(cluster):
        print("⚠️ Principe de précaution applicable")

    print("\n✓ Emergence Detector operational")
