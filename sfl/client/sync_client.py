"""
Synchronous SFL Client - Connect to the Synthetic Field.

Usage:
    from sfl import SFLClient, Report
    import torch

    client = SFLClient("http://localhost:8420")

    # Register a pattern
    pattern_id = client.register(param_shape=(256,))

    # Report activity
    client.report(Report(
        nh_id=pattern_id,
        state=torch.randn(32),
        action=torch.randn(8),
        cost=torch.tensor([0.1]),
        outcome=torch.randn(16)
    ))

    # Pull gradient
    gradient = client.pull_gradient(pattern_id)
"""

from typing import Dict, List, Optional, Tuple, Any
import httpx
import torch

from sfl.core.types import (
    Report,
    GradientPacket,
    AllocSignal,
    UncertaintyBundle,
    ContributionScore,
    PatternInfo,
    LifecycleEvent,
)


class SFLClient:
    """
    Synchronous client for the Synthetic Field Layer.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8420",
        timeout: float = 30.0,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the SFL client.

        Args:
            base_url: Base URL of the SFL server
            timeout: Request timeout in seconds
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers=headers,
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        """Close the client."""
        self._client.close()

    # === Pattern Management ===

    def register(
        self,
        param_shape: Tuple[int, ...],
        lineage: Optional[List[bytes]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        """
        Register a new pattern in the field.

        Args:
            param_shape: Shape of pattern parameters
            lineage: List of ancestor pattern IDs
            metadata: Optional metadata

        Returns:
            Pattern ID (nh_id)
        """
        response = self._client.post(
            "/v1/patterns",
            json={
                "param_shape": list(param_shape),
                "lineage": [l.hex() for l in (lineage or [])],
                "metadata": metadata or {},
            },
        )
        response.raise_for_status()
        data = response.json()
        return bytes.fromhex(data["nh_id"])

    def get_pattern(self, nh_id: bytes) -> PatternInfo:
        """Get pattern information."""
        response = self._client.get(f"/v1/patterns/{nh_id.hex()}")
        response.raise_for_status()
        return PatternInfo.from_dict(response.json())

    def list_patterns(self, status: Optional[str] = None) -> List[PatternInfo]:
        """List all patterns."""
        params = {}
        if status:
            params["status"] = status
        response = self._client.get("/v1/patterns", params=params)
        response.raise_for_status()
        return [PatternInfo.from_dict(p) for p in response.json()["patterns"]]

    def fork(self, nh_id: bytes, n_children: int = 2) -> List[bytes]:
        """Fork a pattern into multiple children."""
        response = self._client.post(
            f"/v1/patterns/{nh_id.hex()}/fork",
            json={"n_children": n_children},
        )
        response.raise_for_status()
        data = response.json()
        return [bytes.fromhex(c) for c in data["children"]]

    def merge(self, pattern_ids: List[bytes]) -> bytes:
        """Merge multiple patterns into one."""
        response = self._client.post(
            "/v1/patterns/merge",
            json={"pattern_ids": [p.hex() for p in pattern_ids]},
        )
        response.raise_for_status()
        data = response.json()
        return bytes.fromhex(data["merged_id"])

    def freeze(self, nh_id: bytes) -> None:
        """Freeze a pattern."""
        response = self._client.post(f"/v1/patterns/{nh_id.hex()}/freeze")
        response.raise_for_status()

    def dissolve(self, nh_id: bytes) -> None:
        """Dissolve a pattern permanently."""
        response = self._client.delete(f"/v1/patterns/{nh_id.hex()}")
        response.raise_for_status()

    # === Reports ===

    def report(self, report: Report) -> None:
        """
        Submit a report to the field.

        Args:
            report: Report containing state, action, cost, outcome
        """
        response = self._client.post("/v1/reports", json=report.to_dict())
        response.raise_for_status()

    def report_batch(self, reports: List[Report]) -> None:
        """Submit multiple reports at once."""
        response = self._client.post(
            "/v1/reports/batch",
            json={"reports": [r.to_dict() for r in reports]},
        )
        response.raise_for_status()

    # === Gradients ===

    def pull_gradient(self, nh_id: bytes) -> GradientPacket:
        """
        Pull the current gradient for a pattern.

        Args:
            nh_id: Pattern ID

        Returns:
            GradientPacket with gradient, allocation, and uncertainty
        """
        response = self._client.get(f"/v1/gradients/{nh_id.hex()}")
        response.raise_for_status()
        return GradientPacket.from_dict(response.json())

    # === Contributions ===

    def get_contribution(self, nh_id: bytes) -> ContributionScore:
        """Get contribution score for a pattern."""
        response = self._client.get(f"/v1/contributions/{nh_id.hex()}")
        response.raise_for_status()
        return ContributionScore.from_dict(response.json())

    # === Field Stats ===

    def get_stats(self) -> Dict[str, Any]:
        """Get field statistics."""
        response = self._client.get("/v1/stats")
        response.raise_for_status()
        return response.json()

    def health(self) -> Dict[str, Any]:
        """Check server health."""
        response = self._client.get("/health")
        response.raise_for_status()
        return response.json()

    # === Convenience Methods ===

    def report_and_pull(
        self,
        nh_id: bytes,
        state: torch.Tensor,
        action: torch.Tensor,
        cost: torch.Tensor,
        outcome: torch.Tensor,
    ) -> GradientPacket:
        """
        Submit a report and immediately pull the gradient.

        This is the most common pattern for real-time agents.
        """
        report = Report(
            nh_id=nh_id,
            state=state,
            action=action,
            cost=cost,
            outcome=outcome,
        )
        self.report(report)
        return self.pull_gradient(nh_id)

    def create_pattern_with_gradient(
        self,
        param_shape: Tuple[int, ...],
        initial_state: torch.Tensor,
        initial_action: torch.Tensor,
        metadata: Optional[Dict] = None,
    ) -> Tuple[bytes, GradientPacket]:
        """
        Register a pattern and get initial gradient in one call.
        """
        nh_id = self.register(param_shape=param_shape, metadata=metadata)

        # Submit initial report
        report = Report(
            nh_id=nh_id,
            state=initial_state,
            action=initial_action,
            cost=torch.tensor([0.0]),
            outcome=torch.zeros(16),
        )
        self.report(report)

        gradient = self.pull_gradient(nh_id)
        return nh_id, gradient
