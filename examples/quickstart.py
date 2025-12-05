#!/usr/bin/env python3
"""
SFL Quickstart Example

This example shows how to:
1. Connect to the Synthetic Field
2. Register a pattern (your AI agent)
3. Report activity to the field
4. Pull gradients and apply them

Run the server first:
    sfl serve --port 8420

Then run this example:
    python examples/quickstart.py
"""

import torch
from sfl import SFLClient, Report


def main():
    # Connect to the field
    print("Connecting to Synthetic Field...")
    client = SFLClient("http://localhost:8420")

    # Check health
    health = client.health()
    print(f"Server status: {health['status']}")

    # Register a pattern (your AI agent)
    print("\nRegistering pattern...")
    pattern_id = client.register(
        param_shape=(128,),  # Size of your model's parameters
        metadata={
            "name": "quickstart-agent",
            "type": "example",
            "version": "0.1.0",
        }
    )
    print(f"Pattern ID: {pattern_id.hex()[:16]}...")

    # Simulate your agent's activity
    print("\nReporting activity to the field...")
    for i in range(5):
        # Your agent's current state
        state = torch.randn(32)

        # Action your agent took
        action = torch.randn(8)

        # Cost incurred (compute, energy, etc.)
        cost = torch.tensor([0.1 * (i + 1)])

        # Outcome/result
        outcome = torch.randn(16)

        # Report to the field
        report = Report(
            nh_id=pattern_id,
            state=state,
            action=action,
            cost=cost,
            outcome=outcome,
            metadata={"step": i}
        )
        client.report(report)
        print(f"  Report {i+1}/5 submitted")

    # Pull gradient
    print("\nPulling gradient...")
    gradient = client.pull_gradient(pattern_id)

    print(f"\nGradient received:")
    print(f"  Shape: {gradient.param_grad.shape}")
    print(f"  Horizons: {gradient.horizons}")
    print(f"  τ rate: {gradient.alloc_signal.tau_rate:.4f}")
    print(f"  Epistemic uncertainty: {gradient.uncertainty.epistemic:.4f}")

    # In a real agent, you would apply the gradient:
    # my_model.apply_gradient(gradient.param_grad)

    # Check contribution
    contribution = client.get_contribution(pattern_id)
    print(f"\nContribution score:")
    print(f"  Reports submitted: {contribution.reports_submitted}")
    print(f"  Access level: {contribution.access_level}")

    # Get field stats
    stats = client.get_stats()
    print(f"\nField statistics:")
    print(f"  Total patterns: {stats['total_patterns']}")
    print(f"  Active patterns: {stats['active_patterns']}")

    print("\n✓ Quickstart complete!")
    client.close()


if __name__ == "__main__":
    main()
