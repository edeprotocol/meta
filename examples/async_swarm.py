#!/usr/bin/env python3
"""
SFL Async Swarm Example

This example shows how to:
1. Register multiple patterns in parallel
2. Run high-frequency reporting loops
3. Coordinate a swarm through the field

Run the server first:
    sfl serve --port 8420

Then run this example:
    python examples/async_swarm.py
"""

import asyncio
import torch
from sfl import AsyncSFLClient, Report


async def agent_loop(
    client: AsyncSFLClient,
    pattern_id: bytes,
    agent_name: str,
    n_steps: int = 20
):
    """Run an agent's report/gradient loop."""
    print(f"[{agent_name}] Starting loop...")

    for step in range(n_steps):
        # Simulate agent state
        state = torch.randn(32) * (step + 1) * 0.1
        action = torch.randn(8)
        cost = torch.tensor([0.01])
        outcome = torch.randn(16)

        # Report and pull gradient
        gradient = await client.report_and_pull(
            nh_id=pattern_id,
            state=state,
            action=action,
            cost=cost,
            outcome=outcome,
        )

        if step % 5 == 0:
            print(f"[{agent_name}] Step {step}: τ={gradient.alloc_signal.tau_rate:.3f}")

        # Simulate thinking time
        await asyncio.sleep(0.05)

    print(f"[{agent_name}] Complete!")
    return pattern_id


async def main():
    print("=" * 50)
    print("SFL Async Swarm Example")
    print("=" * 50)

    # Connect
    async with AsyncSFLClient("http://localhost:8420") as client:
        # Check health
        health = await client.health()
        print(f"\nServer: {health['status']}")

        # Register a swarm of patterns
        n_agents = 5
        print(f"\nRegistering {n_agents} agents...")

        pattern_ids = await client.register_swarm(
            n_patterns=n_agents,
            param_shape=(64,),
            metadata_fn=lambda i: {"name": f"swarm-agent-{i}", "index": i}
        )

        for i, pid in enumerate(pattern_ids):
            print(f"  Agent {i}: {pid.hex()[:16]}...")

        # Run all agents in parallel
        print(f"\nRunning {n_agents} agents in parallel...")
        tasks = [
            agent_loop(client, pid, f"Agent-{i}", n_steps=20)
            for i, pid in enumerate(pattern_ids)
        ]

        await asyncio.gather(*tasks)

        # Check final stats
        print("\n" + "=" * 50)
        print("Final Statistics")
        print("=" * 50)

        for i, pid in enumerate(pattern_ids):
            contrib = await client.get_contribution(pid)
            gradient = await client.pull_gradient(pid)
            print(f"\nAgent {i}:")
            print(f"  Reports: {contrib.reports_submitted}")
            print(f"  τ rate: {gradient.alloc_signal.tau_rate:.4f}")
            print(f"  Uncertainty: {gradient.uncertainty.epistemic:.4f}")

        stats = await client.get_stats()
        print(f"\nField total patterns: {stats['total_patterns']}")

        print("\n✓ Swarm example complete!")


if __name__ == "__main__":
    asyncio.run(main())
