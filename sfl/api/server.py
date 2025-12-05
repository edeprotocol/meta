"""
SFL API Server - FastAPI server for the Synthetic Field Layer.

Run with:
    uvicorn sfl.api.server:app --host 0.0.0.0 --port 8420

Or use the CLI:
    sfl serve --port 8420
"""

from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch

from sfl.core.kernel import SyntheticFieldKernel
from sfl.core.types import Report as CoreReport


# === Pydantic Models ===


class RegisterRequest(BaseModel):
    param_shape: List[int]
    lineage: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RegisterResponse(BaseModel):
    nh_id: str
    created_at: int


class ReportRequest(BaseModel):
    nh_id: str
    state: List[float]
    action: List[float]
    cost: List[float]
    outcome: List[float]
    timestamp: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BatchReportRequest(BaseModel):
    reports: List[ReportRequest]


class ForkRequest(BaseModel):
    n_children: int = 2


class ForkResponse(BaseModel):
    children: List[str]


class MergeRequest(BaseModel):
    pattern_ids: List[str]


class MergeResponse(BaseModel):
    merged_id: str


class UncertaintyResponse(BaseModel):
    epistemic: float
    aleatoric: float
    model: float
    adversarial: float
    scalar: float


class AllocSignalResponse(BaseModel):
    tau_rate: float
    allowed_envs: List[str]
    compliance_window: int
    non_compliance_penalty: float


class GradientResponse(BaseModel):
    nh_id: str
    param_grad: List[List[float]]
    horizons: List[float]
    alloc_signal: AllocSignalResponse
    uncertainty: UncertaintyResponse
    critic_ids: List[str]
    timestamp: int


class ContributionResponse(BaseModel):
    reports_submitted: int
    gradient_utility: float
    data_uniqueness: float
    access_level: int


class PatternResponse(BaseModel):
    nh_id: str
    param_shape: List[int]
    lineage: List[str]
    created_at: int
    tau_rate: float
    status: str
    metadata: Dict[str, Any]


class PatternListResponse(BaseModel):
    patterns: List[PatternResponse]
    total: int


class StatsResponse(BaseModel):
    total_patterns: int
    active_patterns: int
    frozen_patterns: int
    total_reports: int
    total_lifecycle_events: int
    memory_entries: int


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float


# === Global State ===

kernel: Optional[SyntheticFieldKernel] = None
start_time: float = 0


# === App Factory ===


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global kernel, start_time
    import time

    # Initialize kernel
    kernel = SyntheticFieldKernel({
        "horizons": [1.0, 10.0],
        "n_critics": 2,
        "critic_hidden_dim": 256,
        "memory_path": "./field_memory",
    })
    start_time = time.time()

    yield

    # Cleanup
    kernel = None


def create_app(config: Optional[Dict] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    global kernel

    app = FastAPI(
        title="Synthetic Field Layer",
        description="The Economic Operating System for AGI/ASI",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


# Create default app
app = create_app()


# === Health ===


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """Check server health."""
    import time

    return HealthResponse(
        status="healthy",
        version="0.1.0",
        uptime_seconds=time.time() - start_time,
    )


# === Patterns ===


@app.post("/v1/patterns", response_model=RegisterResponse, tags=["Patterns"])
async def register_pattern(request: RegisterRequest):
    """Register a new pattern in the field."""
    if kernel is None:
        raise HTTPException(status_code=503, detail="Kernel not initialized")

    try:
        nh_id = kernel.register(
            param_shape=tuple(request.param_shape),
            lineage=[bytes.fromhex(l) for l in request.lineage],
            metadata=request.metadata,
        )
        pattern = kernel.get_pattern(nh_id)
        return RegisterResponse(
            nh_id=nh_id.hex(),
            created_at=pattern.created_at,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/v1/patterns", response_model=PatternListResponse, tags=["Patterns"])
async def list_patterns(status: Optional[str] = Query(None)):
    """List all patterns."""
    if kernel is None:
        raise HTTPException(status_code=503, detail="Kernel not initialized")

    patterns = kernel.list_patterns(status=status)
    return PatternListResponse(
        patterns=[
            PatternResponse(
                nh_id=p.nh_id.hex(),
                param_shape=list(p.param_shape),
                lineage=[l.hex() for l in p.lineage],
                created_at=p.created_at,
                tau_rate=p.tau_rate,
                status=p.status,
                metadata=p.metadata,
            )
            for p in patterns
        ],
        total=len(patterns),
    )


@app.get("/v1/patterns/{nh_id}", response_model=PatternResponse, tags=["Patterns"])
async def get_pattern(nh_id: str):
    """Get pattern information."""
    if kernel is None:
        raise HTTPException(status_code=503, detail="Kernel not initialized")

    try:
        pattern = kernel.get_pattern(bytes.fromhex(nh_id))
        return PatternResponse(
            nh_id=pattern.nh_id.hex(),
            param_shape=list(pattern.param_shape),
            lineage=[l.hex() for l in pattern.lineage],
            created_at=pattern.created_at,
            tau_rate=pattern.tau_rate,
            status=pattern.status,
            metadata=pattern.metadata,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/v1/patterns/{nh_id}/fork", response_model=ForkResponse, tags=["Patterns"])
async def fork_pattern(nh_id: str, request: ForkRequest):
    """Fork a pattern into multiple children."""
    if kernel is None:
        raise HTTPException(status_code=503, detail="Kernel not initialized")

    try:
        children = kernel.fork(bytes.fromhex(nh_id), n_children=request.n_children)
        return ForkResponse(children=[c.hex() for c in children])
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/v1/patterns/merge", response_model=MergeResponse, tags=["Patterns"])
async def merge_patterns(request: MergeRequest):
    """Merge multiple patterns into one."""
    if kernel is None:
        raise HTTPException(status_code=503, detail="Kernel not initialized")

    try:
        merged_id = kernel.merge([bytes.fromhex(p) for p in request.pattern_ids])
        return MergeResponse(merged_id=merged_id.hex())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/v1/patterns/{nh_id}/freeze", tags=["Patterns"])
async def freeze_pattern(nh_id: str):
    """Freeze a pattern."""
    if kernel is None:
        raise HTTPException(status_code=503, detail="Kernel not initialized")

    try:
        kernel.freeze(bytes.fromhex(nh_id))
        return {"status": "frozen"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/v1/patterns/{nh_id}", tags=["Patterns"])
async def dissolve_pattern(nh_id: str):
    """Dissolve a pattern permanently."""
    if kernel is None:
        raise HTTPException(status_code=503, detail="Kernel not initialized")

    try:
        kernel.dissolve(bytes.fromhex(nh_id))
        return {"status": "dissolved"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# === Reports ===


@app.post("/v1/reports", tags=["Reports"])
async def submit_report(request: ReportRequest):
    """Submit a report to the field."""
    if kernel is None:
        raise HTTPException(status_code=503, detail="Kernel not initialized")

    try:
        report = CoreReport(
            nh_id=bytes.fromhex(request.nh_id),
            state=torch.tensor(request.state),
            action=torch.tensor(request.action),
            cost=torch.tensor(request.cost),
            outcome=torch.tensor(request.outcome),
            timestamp=request.timestamp,
            metadata=request.metadata,
        )
        kernel.report(report)
        return {"status": "accepted"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/v1/reports/batch", tags=["Reports"])
async def submit_batch_reports(request: BatchReportRequest):
    """Submit multiple reports at once."""
    if kernel is None:
        raise HTTPException(status_code=503, detail="Kernel not initialized")

    errors = []
    for i, r in enumerate(request.reports):
        try:
            report = CoreReport(
                nh_id=bytes.fromhex(r.nh_id),
                state=torch.tensor(r.state),
                action=torch.tensor(r.action),
                cost=torch.tensor(r.cost),
                outcome=torch.tensor(r.outcome),
                timestamp=r.timestamp,
                metadata=r.metadata,
            )
            kernel.report(report)
        except Exception as e:
            errors.append({"index": i, "error": str(e)})

    if errors:
        return {"status": "partial", "errors": errors, "accepted": len(request.reports) - len(errors)}
    return {"status": "accepted", "count": len(request.reports)}


# === Gradients ===


@app.get("/v1/gradients/{nh_id}", response_model=GradientResponse, tags=["Gradients"])
async def pull_gradient(nh_id: str):
    """Pull the current gradient for a pattern."""
    if kernel is None:
        raise HTTPException(status_code=503, detail="Kernel not initialized")

    try:
        gradient = kernel.pull_gradient(bytes.fromhex(nh_id))
        return GradientResponse(
            nh_id=gradient.nh_id.hex(),
            param_grad=gradient.param_grad.tolist(),
            horizons=gradient.horizons,
            alloc_signal=AllocSignalResponse(
                tau_rate=gradient.alloc_signal.tau_rate,
                allowed_envs=[e.hex() for e in gradient.alloc_signal.allowed_envs],
                compliance_window=gradient.alloc_signal.compliance_window,
                non_compliance_penalty=gradient.alloc_signal.non_compliance_penalty,
            ),
            uncertainty=UncertaintyResponse(
                epistemic=gradient.uncertainty.epistemic,
                aleatoric=gradient.uncertainty.aleatoric,
                model=gradient.uncertainty.model,
                adversarial=gradient.uncertainty.adversarial,
                scalar=gradient.uncertainty.scalar,
            ),
            critic_ids=[c.hex() for c in gradient.critic_ids],
            timestamp=gradient.timestamp,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# === Contributions ===


@app.get("/v1/contributions/{nh_id}", response_model=ContributionResponse, tags=["Contributions"])
async def get_contribution(nh_id: str):
    """Get contribution score for a pattern."""
    if kernel is None:
        raise HTTPException(status_code=503, detail="Kernel not initialized")

    try:
        contribution = kernel.get_contribution(bytes.fromhex(nh_id))
        return ContributionResponse(
            reports_submitted=contribution.reports_submitted,
            gradient_utility=contribution.gradient_utility,
            data_uniqueness=contribution.data_uniqueness,
            access_level=contribution.access_level,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# === Stats ===


@app.get("/v1/stats", response_model=StatsResponse, tags=["Stats"])
async def get_stats():
    """Get field statistics."""
    if kernel is None:
        raise HTTPException(status_code=503, detail="Kernel not initialized")

    stats = kernel.get_stats()
    return StatsResponse(**stats)


# === Entry point ===


def run_server(host: str = "0.0.0.0", port: int = 8420, workers: int = 1):
    """Run the server."""
    import uvicorn

    uvicorn.run(
        "sfl.api.server:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
    )


if __name__ == "__main__":
    run_server()
