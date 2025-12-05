"""
Asynchronous SFL Client - High-performance connection to the Synthetic Field.

Usage:
    import asyncio
    from sfl import AsyncSFLClient, Report
    import torch

    async def main():
        async with AsyncSFLClient("http://localhost:8420") as client:
            pattern_id = await client.register(param_shape=(256,))

            await client.report(Report(
                nh_id=pattern_id,
                state=torch.randn(32),
                action=torch.randn(8),
                cost=torch.tensor([0.1]),
                outcome=torch.randn(16)
            ))

            gradient = await client.pull_gradient(pattern_id)

    asyncio.run(main())
"""

import asyncio
from typing import AsyncIterator, Dict, List, Optional, Tuple, Any
import httpx
import torch

from sfl.core.types import (
    Report,
    GradientPacket,
    ContributionScore,
    PatternInfo,
)


class AsyncSFLClient:
    """
    Asynchronous client for the Synthetic Field Layer.

    Supports high-frequency reporting and streaming.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8420",
        timeout: float = 30.0,
        api_key: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key

        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            headers=headers,
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    async def close(self):
        """Close the client."""
        await self._client.aclose()

    # === Pattern Management ===

    async def register(
        self,
        param_shape: Tuple[int, ...],
        lineage: Optional[List[bytes]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bytes:
        """Register a new pattern in the field."""
        response = await self._client.post(
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

    async def get_pattern(self, nh_id: bytes) -> PatternInfo:
        """Get pattern information."""
        response = await self._client.get(f"/v1/patterns/{nh_id.hex()}")
        response.raise_for_status()
        return PatternInfo.from_dict(response.json())

    async def list_patterns(self, status: Optional[str] = None) -> List[PatternInfo]:
        """List all patterns."""
        params = {}
        if status:
            params["status"] = status
        response = await self._client.get("/v1/patterns", params=params)
        response.raise_for_status()
        return [PatternInfo.from_dict(p) for p in response.json()["patterns"]]

    async def fork(self, nh_id: bytes, n_children: int = 2) -> List[bytes]:
        """Fork a pattern into multiple children."""
        response = await self._client.post(
            f"/v1/patterns/{nh_id.hex()}/fork",
            json={"n_children": n_children},
        )
        response.raise_for_status()
        data = response.json()
        return [bytes.fromhex(c) for c in data["children"]]

    async def merge(self, pattern_ids: List[bytes]) -> bytes:
        """Merge multiple patterns into one."""
        response = await self._client.post(
            "/v1/patterns/merge",
            json={"pattern_ids": [p.hex() for p in pattern_ids]},
        )
        response.raise_for_status()
        data = response.json()
        return bytes.fromhex(data["merged_id"])

    async def freeze(self, nh_id: bytes) -> None:
        """Freeze a pattern."""
        response = await self._client.post(f"/v1/patterns/{nh_id.hex()}/freeze")
        response.raise_for_status()

    async def dissolve(self, nh_id: bytes) -> None:
        """Dissolve a pattern permanently."""
        response = await self._client.delete(f"/v1/patterns/{nh_id.hex()}")
        response.raise_for_status()

    # === Reports ===

    async def report(self, report: Report) -> None:
        """Submit a report to the field."""
        response = await self._client.post("/v1/reports", json=report.to_dict())
        response.raise_for_status()

    async def report_batch(self, reports: List[Report]) -> None:
        """Submit multiple reports at once."""
        response = await self._client.post(
            "/v1/reports/batch",
            json={"reports": [r.to_dict() for r in reports]},
        )
        response.raise_for_status()

    # === Gradients ===

    async def pull_gradient(self, nh_id: bytes) -> GradientPacket:
        """Pull the current gradient for a pattern."""
        response = await self._client.get(f"/v1/gradients/{nh_id.hex()}")
        response.raise_for_status()
        return GradientPacket.from_dict(response.json())

    # === Contributions ===

    async def get_contribution(self, nh_id: bytes) -> ContributionScore:
        """Get contribution score for a pattern."""
        response = await self._client.get(f"/v1/contributions/{nh_id.hex()}")
        response.raise_for_status()
        return ContributionScore.from_dict(response.json())

    # === Field Stats ===

    async def get_stats(self) -> Dict[str, Any]:
        """Get field statistics."""
        response = await self._client.get("/v1/stats")
        response.raise_for_status()
        return response.json()

    async def health(self) -> Dict[str, Any]:
        """Check server health."""
        response = await self._client.get("/health")
        response.raise_for_status()
        return response.json()

    # === High-Frequency Methods ===

    async def report_and_pull(
        self,
        nh_id: bytes,
        state: torch.Tensor,
        action: torch.Tensor,
        cost: torch.Tensor,
        outcome: torch.Tensor,
    ) -> GradientPacket:
        """Submit report and pull gradient in one call."""
        report = Report(
            nh_id=nh_id,
            state=state,
            action=action,
            cost=cost,
            outcome=outcome,
        )
        await self.report(report)
        return await self.pull_gradient(nh_id)

    async def report_loop(
        self,
        nh_id: bytes,
        state_fn,
        action_fn,
        cost_fn,
        outcome_fn,
        frequency_hz: float = 100,
        callback=None,
    ) -> None:
        """
        Continuous reporting loop at specified frequency.

        Args:
            nh_id: Pattern ID
            state_fn: Function returning current state tensor
            action_fn: Function returning current action tensor
            cost_fn: Function returning current cost tensor
            outcome_fn: Function returning current outcome tensor
            frequency_hz: Reporting frequency in Hz
            callback: Optional callback(gradient) called after each pull
        """
        interval = 1.0 / frequency_hz

        while True:
            start = asyncio.get_event_loop().time()

            report = Report(
                nh_id=nh_id,
                state=state_fn(),
                action=action_fn(),
                cost=cost_fn(),
                outcome=outcome_fn(),
            )
            await self.report(report)

            gradient = await self.pull_gradient(nh_id)

            if callback:
                callback(gradient)

            elapsed = asyncio.get_event_loop().time() - start
            sleep_time = max(0, interval - elapsed)
            await asyncio.sleep(sleep_time)

    # === Parallel Operations ===

    async def register_swarm(
        self,
        n_patterns: int,
        param_shape: Tuple[int, ...],
        metadata_fn=None,
    ) -> List[bytes]:
        """
        Register multiple patterns in parallel.

        Args:
            n_patterns: Number of patterns to register
            param_shape: Shape for all patterns
            metadata_fn: Optional function(index) -> metadata

        Returns:
            List of pattern IDs
        """
        tasks = []
        for i in range(n_patterns):
            metadata = metadata_fn(i) if metadata_fn else {"index": i}
            tasks.append(self.register(param_shape=param_shape, metadata=metadata))

        return await asyncio.gather(*tasks)

    async def report_swarm(self, reports: List[Report]) -> None:
        """Submit reports for multiple patterns in parallel."""
        tasks = [self.report(r) for r in reports]
        await asyncio.gather(*tasks)

    async def pull_gradients(self, nh_ids: List[bytes]) -> List[GradientPacket]:
        """Pull gradients for multiple patterns in parallel."""
        tasks = [self.pull_gradient(nh_id) for nh_id in nh_ids]
        return await asyncio.gather(*tasks)
