"""
SFL CLI - Command line interface for the Synthetic Field Layer.

Usage:
    sfl serve --port 8420
    sfl status
    sfl patterns list
    sfl pattern <nh_id> info
"""

import typer
from rich.console import Console
from rich.table import Table
from typing import Optional

app = typer.Typer(
    name="sfl",
    help="Synthetic Field Layer - The Economic Operating System for AGI/ASI",
    add_completion=False,
)
console = Console()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8420, "--port", "-p", help="Port to bind to"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of workers"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
):
    """Start the SFL server."""
    import uvicorn

    console.print(f"[bold green]Starting SFL server on {host}:{port}[/bold green]")
    console.print("[dim]Press Ctrl+C to stop[/dim]")

    uvicorn.run(
        "sfl.api.server:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info",
    )


@app.command()
def status(
    url: str = typer.Option("http://localhost:8420", "--url", "-u", help="Server URL"),
):
    """Check server status."""
    import httpx

    try:
        with httpx.Client(timeout=5.0) as client:
            health = client.get(f"{url}/health").json()
            stats = client.get(f"{url}/v1/stats").json()

        console.print("\n[bold green]✓ Server is healthy[/bold green]")
        console.print(f"  Version: {health['version']}")
        console.print(f"  Uptime: {health['uptime_seconds']:.1f}s")

        console.print("\n[bold]Field Statistics:[/bold]")
        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for key, value in stats.items():
            table.add_row(key.replace("_", " ").title(), str(value))

        console.print(table)

    except httpx.ConnectError:
        console.print(f"[bold red]✗ Cannot connect to {url}[/bold red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)


@app.command("patterns")
def list_patterns(
    url: str = typer.Option("http://localhost:8420", "--url", "-u"),
    status_filter: Optional[str] = typer.Option(None, "--status", "-s"),
):
    """List all patterns in the field."""
    import httpx

    try:
        with httpx.Client(timeout=10.0) as client:
            params = {}
            if status_filter:
                params["status"] = status_filter
            response = client.get(f"{url}/v1/patterns", params=params)
            data = response.json()

        table = Table(title=f"Patterns ({data['total']} total)")
        table.add_column("ID", style="cyan", no_wrap=True)
        table.add_column("Status", style="green")
        table.add_column("τ Rate", style="yellow")
        table.add_column("Shape", style="dim")

        for p in data["patterns"]:
            nh_id_short = p["nh_id"][:16] + "..."
            status_style = {
                "active": "[green]active[/green]",
                "frozen": "[blue]frozen[/blue]",
                "dissolved": "[red]dissolved[/red]",
            }.get(p["status"], p["status"])

            table.add_row(
                nh_id_short,
                status_style,
                f"{p['tau_rate']:.2f}",
                str(p["param_shape"]),
            )

        console.print(table)

    except httpx.ConnectError:
        console.print(f"[bold red]✗ Cannot connect to {url}[/bold red]")
        raise typer.Exit(1)


@app.command("register")
def register_pattern(
    shape: str = typer.Argument(..., help="Parameter shape, e.g., '256' or '128,64'"),
    url: str = typer.Option("http://localhost:8420", "--url", "-u"),
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Pattern name"),
):
    """Register a new pattern."""
    import httpx

    try:
        param_shape = [int(x.strip()) for x in shape.split(",")]

        metadata = {}
        if name:
            metadata["name"] = name

        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{url}/v1/patterns",
                json={
                    "param_shape": param_shape,
                    "lineage": [],
                    "metadata": metadata,
                },
            )
            data = response.json()

        console.print(f"[bold green]✓ Pattern registered[/bold green]")
        console.print(f"  ID: {data['nh_id']}")

    except httpx.ConnectError:
        console.print(f"[bold red]✗ Cannot connect to {url}[/bold red]")
        raise typer.Exit(1)


@app.command("gradient")
def get_gradient(
    nh_id: str = typer.Argument(..., help="Pattern ID"),
    url: str = typer.Option("http://localhost:8420", "--url", "-u"),
):
    """Pull gradient for a pattern."""
    import httpx

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(f"{url}/v1/gradients/{nh_id}")
            data = response.json()

        console.print(f"\n[bold]Gradient for {nh_id[:16]}...[/bold]")
        console.print(f"  τ Rate: {data['alloc_signal']['tau_rate']:.4f}")
        console.print(f"  Horizons: {data['horizons']}")
        console.print(f"  Epistemic Uncertainty: {data['uncertainty']['epistemic']:.4f}")
        console.print(f"  Critics: {len(data['critic_ids'])}")

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            console.print(f"[bold red]Pattern not found: {nh_id}[/bold red]")
        else:
            console.print(f"[bold red]Error: {e}[/bold red]")
        raise typer.Exit(1)
    except httpx.ConnectError:
        console.print(f"[bold red]✗ Cannot connect to {url}[/bold red]")
        raise typer.Exit(1)


@app.command("version")
def version():
    """Show version information."""
    from sfl import __version__

    console.print(f"Synthetic Field Layer v{__version__}")
    console.print("The Economic Operating System for AGI/ASI")


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
