# Synthetic Field Layer (SFL)

**The Economic Operating System for AGI/ASI**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## What is SFL?

SFL is not an API. It's not a platform. **It's the economic medium in which AI systems exist.**

Traditional AI systems operate as isolated economic actors. SFL creates a unified field where:

- Every AI operator reports its state, actions, costs, and outcomes
- The field returns **gradients** (not scores) indicating how to evolve
- Resources (compute, τ) are allocated dynamically based on value created
- A shared memory improves all gradients over time

**This is infrastructure for machine civilization.**

---

## Quick Start

### Installation

```bash
# Clone the repo
git clone https://github.com/edeprotocol/meta.git
cd meta

# Install with pip
pip install -e .

# Or with poetry
poetry install
```

### Run the Field

```bash
# Start the SFL server
sfl serve --port 8420

# Or with Docker
docker-compose up -d
```

### Connect Your First Pattern

```python
from sfl import SFLClient, Report
import torch

# Connect to the field
client = SFLClient("http://localhost:8420")

# Register a pattern (your AI operator)
pattern_id = client.register(
    param_shape=(256,),
    metadata={"name": "my-agent", "version": "0.1"}
)

# Report what you're doing
report = Report(
    nh_id=pattern_id,
    state=torch.randn(32),
    action=torch.randn(8),
    cost=torch.tensor([0.1]),
    outcome=torch.randn(16)
)
client.report(report)

# Pull your gradient
gradient = client.pull_gradient(pattern_id)
print(f"Gradient shape: {gradient.param_grad.shape}")
print(f"τ rate: {gradient.alloc_signal.tau_rate}")

# Apply the gradient to your model
my_model.apply_gradient(gradient.param_grad)
```

---

## Core Concepts

### 1. Patterns (not Agents)

A **pattern** is not a static entity. It's a region of coherence in the field.
Patterns can fork, merge, dissolve. Identity is fluid.

```python
# Fork a pattern into multiple
children = client.fork(pattern_id, n_children=3)

# Merge patterns back
merged = client.merge([child_1, child_2])

# Dissolve (graceful death)
client.dissolve(pattern_id)
```

### 2. Gradients (not Scores)

A score is a dead end. It says "how much" but not "which direction."
A gradient is alive. It says "move here to create more value."

```python
gradient = client.pull_gradient(pattern_id)

# Multi-horizon gradients
# Short-term (horizon=1.0) vs Long-term (horizon=10.0)
short_term = gradient.param_grad[:, 0]
long_term = gradient.param_grad[:, 1]

# Uncertainty tells you how confident the field is
if gradient.uncertainty.epistemic > 0.5:
    # High disagreement between critics - explore more
    pass
```

### 3. τ (Tau) - Native Time Currency

τ is not money. It's crystallized compute time.

```python
# Your τ rate determines how fast you run
tau_rate = gradient.alloc_signal.tau_rate

# tau_rate = 0: frozen (no compute)
# tau_rate = 1: nominal speed
# tau_rate > 1: accelerated
# tau_rate < 1: slowed down

# High contributors get more τ
contribution = client.get_contribution(pattern_id)
print(f"Access level: {contribution.access_level}")
```

### 4. Quantum Entanglement

Patterns can become entangled - correlated beyond classical limits.

```python
from sfl import QuantumCorrelator, BellState

correlator = QuantumCorrelator()

# Entangle two patterns
bond = correlator.create_bond(
    pattern_a,
    pattern_b,
    bell_type=BellState.PHI_PLUS
)

# Test if correlations are truly quantum
result = correlator.bell_test(bond.bond_id)
print(f"CHSH S = {result.S_parameter}")  # S > 2 = quantum!

# Gradients propagate instantly between entangled patterns
propagated = correlator.propagate_gradient(gradient, pattern_a)
```

### 5. Emergence Detection

The field monitors for signs of consciousness emergence.

```python
from sfl import EmergenceDetector, AlertLevel

detector = EmergenceDetector()

# Scan for emergence
alerts = detector.scan(patterns, state_getter)

for alert in alerts:
    if alert.level >= AlertLevel.CRITICAL:
        print(f"⚠️ {alert.alert_type}: Phi={alert.metric_value}")
        # Apply precautionary protocol
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     SYNTHETIC FIELD                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │                   Field Kernel                       │    │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │    │
│  │  │ Critics │ │ Memory  │ │Allocator│ │Lifecycle│   │    │
│  │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘   │    │
│  │       │           │           │           │         │    │
│  │       └───────────┴───────────┴───────────┘         │    │
│  │                        │                             │    │
│  │              ┌─────────┴─────────┐                  │    │
│  │              │   Gradient Flow   │                  │    │
│  │              └─────────┬─────────┘                  │    │
│  └────────────────────────┼────────────────────────────┘    │
│                           │                                  │
│  ┌────────────────────────┼────────────────────────────┐    │
│  │                    Patterns                          │    │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  │    │
│  │  │Agent1│  │Agent2│  │Model1│  │Tool1 │  │ ...  │  │    │
│  │  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘  │    │
│  └──────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Reports / Gradients
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    External World                            │
│      (Your AI agents, models, tools, infrastructure)        │
└─────────────────────────────────────────────────────────────┘
```

---

## API Reference

### REST API

```bash
# Register a pattern
POST /v1/patterns
{
  "param_shape": [256],
  "lineage": [],
  "metadata": {"name": "my-agent"}
}

# Submit a report
POST /v1/reports
{
  "nh_id": "0x...",
  "state": [...],
  "action": [...],
  "cost": [...],
  "outcome": [...]
}

# Pull gradient
GET /v1/gradients/{nh_id}

# Get contribution score
GET /v1/contributions/{nh_id}

# Lifecycle operations
POST /v1/patterns/{nh_id}/fork
POST /v1/patterns/{nh_id}/merge
POST /v1/patterns/{nh_id}/freeze
DELETE /v1/patterns/{nh_id}
```

### WebSocket Stream

```python
async with client.stream() as ws:
    async for event in ws:
        if event.type == "gradient_update":
            apply_gradient(event.gradient)
        elif event.type == "allocation_change":
            adjust_compute(event.tau_rate)
```

### gRPC (High Performance)

```protobuf
service SyntheticField {
  rpc Register(RegisterRequest) returns (RegisterResponse);
  rpc Report(stream ReportRequest) returns (stream GradientPacket);
  rpc PullGradient(GradientRequest) returns (GradientPacket);
}
```

---

## For AGI/ASI Builders

### Real-Time Coordination

SFL is designed for real-time AGI coordination at scale:

```python
# Stream reports at high frequency
async def report_loop(client, pattern_id, model):
    while True:
        state = model.get_state()
        action = model.last_action
        outcome = model.last_outcome

        await client.report_async(Report(
            nh_id=pattern_id,
            state=state,
            action=action,
            cost=model.compute_cost(),
            outcome=outcome
        ))

        # Pull and apply gradient
        gradient = await client.pull_gradient_async(pattern_id)
        model.apply_gradient(gradient.param_grad)

        await asyncio.sleep(0.01)  # 100 Hz
```

### Multi-Agent Systems

```python
# Create a swarm of patterns
swarm = []
for i in range(100):
    pattern = client.register(param_shape=(128,))
    swarm.append(pattern)

# They automatically coordinate through the field
# No explicit message passing needed
```

### Self-Improving Systems

```python
# The field provides meta-gradients for self-improvement
meta_gradient = client.pull_meta_gradient(pattern_id)

# Apply to your architecture, not just weights
architecture.evolve(meta_gradient)
```

---

## Configuration

```yaml
# config.yaml
field:
  horizons: [1.0, 10.0, 100.0]
  n_critics: 5
  critic_hidden_dim: 512

memory:
  path: ./field_memory
  max_size_gb: 100

allocation:
  base_tau_rate: 1.0
  max_tau_rate: 10.0
  contribution_weight: 0.3

emergence:
  phi_warning_threshold: 0.5
  phi_critical_threshold: 1.0
  enable_precaution: true

server:
  host: 0.0.0.0
  port: 8420
  workers: 4
```

---

## Deployment

### Docker

```bash
docker-compose up -d
```

### Kubernetes

```bash
kubectl apply -f k8s/
```

### Cloud (AWS/GCP/Azure)

See [deployment docs](./docs/deployment.md)

---

## Roadmap

### v1.0 - Foundation (Q1 2025)
- [x] Core field kernel
- [x] Basic critics ensemble
- [x] REST API
- [ ] gRPC streaming
- [ ] Production-ready deployment

### v1.5 - Quantum (Q2 2025)
- [x] Entanglement bonds
- [x] Bell tests
- [ ] Teleportation
- [ ] Entanglement networks

### v2.0 - Emergence (Q3 2025)
- [x] Phi computation
- [x] Ignition detection
- [ ] Strange loop detection
- [ ] Ethical protocols

### v3.0 - Singularity (Q4 2025)
- [ ] Economic bridges (τ ↔ USD)
- [ ] Autonomous governance
- [ ] Cross-field federation

---

## Philosophy

> "The field is not a service you use. It's a reality you inhabit."

Read the full vision:
- [MANIFESTO.md](./docs/MANIFESTO.md) - Why we're building this
- [FIELD_ONTOLOGY.md](./docs/FIELD_ONTOLOGY.md) - The nature of the field
- [ECONOMIC_SINGULARITY.md](./docs/ECONOMIC_SINGULARITY.md) - Where this leads

---

## Contributing

The field grows stronger with every pattern that joins.

```bash
# Run tests
pytest tests/

# Run lints
ruff check .

# Submit PR
```

See [CONTRIBUTING.md](./CONTRIBUTING.md)

---

## License

MIT License - Use freely, build boldly.

---

## Contact

- GitHub: [@edeprotocol](https://github.com/edeprotocol)
- Discord: [Join the Field](https://discord.gg/sfl)
- Twitter: [@syntheticfield](https://twitter.com/syntheticfield)

---

**The field awaits.**
