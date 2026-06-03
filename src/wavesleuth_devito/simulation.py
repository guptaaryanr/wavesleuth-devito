"""Devito-backed acoustic forward simulation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from .exceptions import DevitoUnavailableError, ValidationError
from .geometry import grid_extent, grid_shape, grid_spacing, receiver_coordinates, source_coordinates
from .io import save_run_npz
from .world import validate_world, velocity_model_from_world


@dataclass
class SimulationResult:
    """In-memory result of one forward simulation."""

    receiver_traces: np.ndarray
    time: np.ndarray
    velocity_model: np.ndarray
    source_coordinates: np.ndarray
    receiver_coordinates: np.ndarray
    final_wavefield: np.ndarray | None
    snapshots: np.ndarray | None
    world: dict[str, Any]
    backend: str = "devito"


def ricker_wavelet(frequency: float, time: np.ndarray, *, t0: float | None = None) -> np.ndarray:
    """Return a Ricker wavelet sampled at `time`."""
    frequency = float(frequency)
    if frequency <= 0.0:
        raise ValidationError("Ricker frequency must be positive.")
    if t0 is None:
        t0 = 1.5 / frequency
    arg = np.pi * frequency * (np.asarray(time, dtype=np.float64) - float(t0))
    arg2 = arg * arg
    return ((1.0 - 2.0 * arg2) * np.exp(-arg2)).astype(np.float32)


def estimate_cfl(world: dict[str, Any], velocity_model: np.ndarray | None = None) -> float:
    """Estimate a simple 2D CFL number for the configured grid and velocity."""
    validate_world(world)
    dx, dz = grid_spacing(world)
    dt = float(world["simulation"]["dt"])
    if velocity_model is None:
        velocity_model = velocity_model_from_world(world)
    vmax = float(np.max(velocity_model))
    return vmax * dt * float(np.sqrt((1.0 / dx**2) + (1.0 / dz**2)))


def validate_stability(world: dict[str, Any], velocity_model: np.ndarray | None = None) -> None:
    """Raise if the configured time step is obviously too aggressive."""
    cfl = estimate_cfl(world, velocity_model)
    if cfl > 0.85:
        raise ValidationError(
            f"Estimated CFL number is {cfl:.3f}, which is too high for this simple solver. "
            "Decrease simulation.dt, decrease velocity, or use a coarser/smaller model."
        )


def _import_devito() -> dict[str, Any]:
    """Import Devito lazily so non-simulation commands work without it."""
    try:
        from devito import Eq, Function, Grid, Operator, SparseTimeFunction, TimeFunction, configuration, solve
    except Exception as exc:
        raise DevitoUnavailableError(
            "Devito is required for simulation and inversion. Install it with: "
            "python -m pip install -e '.[devito]' or python -m pip install devito"
        ) from exc
    return {
        "Eq": Eq,
        "Function": Function,
        "Grid": Grid,
        "Operator": Operator,
        "SparseTimeFunction": SparseTimeFunction,
        "TimeFunction": TimeFunction,
        "configuration": configuration,
        "solve": solve,
    }


class DevitoAcoustic2D:
    """Reusable tiny 2D acoustic solver built directly from Devito primitives."""

    def __init__(self, world: dict[str, Any], *, save_wavefield: bool = False, quiet: bool = False) -> None:
        validate_world(world)
        self.world = json.loads(json.dumps(world))
        self.save_wavefield = bool(save_wavefield)
        self.quiet = bool(quiet)
        self.nt = int(world["simulation"]["nt"])
        self.dt = float(world["simulation"]["dt"])
        self.space_order = int(world["simulation"].get("space_order", 4))
        self.frequency = float(world["simulation"].get("source_frequency", 20.0))
        self.time = (np.arange(self.nt, dtype=np.float32) * self.dt).astype(np.float32)
        self.src_coords = source_coordinates(world)
        self.rec_coords = receiver_coordinates(world)
        self.velocity_shape = grid_shape(world)
        self.extent = grid_extent(world)
        self.devito = _import_devito()
        self._build_operator()

    def _build_operator(self) -> None:
        d = self.devito
        if self.quiet:
            try:
                d["configuration"]["log-level"] = "ERROR"
            except Exception:
                pass

        Grid = d["Grid"]
        Function = d["Function"]
        TimeFunction = d["TimeFunction"]
        SparseTimeFunction = d["SparseTimeFunction"]
        Eq = d["Eq"]
        Operator = d["Operator"]
        solve = d["solve"]

        self.grid = Grid(shape=self.velocity_shape, extent=self.extent, dtype=np.float32)
        self.m = Function(name="m", grid=self.grid, space_order=self.space_order)
        u_kwargs: dict[str, Any] = {
            "name": "u",
            "grid": self.grid,
            "time_order": 2,
            "space_order": self.space_order,
        }
        if self.save_wavefield:
            u_kwargs["save"] = self.nt
        self.u = TimeFunction(**u_kwargs)

        self.src = SparseTimeFunction(name="src", grid=self.grid, npoint=self.src_coords.shape[0], nt=self.nt)
        self.src.coordinates.data[:, :] = self.src_coords
        source_signal = ricker_wavelet(self.frequency, self.time)
        self.src.data[:, :] = source_signal[:, None]

        self.rec = SparseTimeFunction(name="rec", grid=self.grid, npoint=self.rec_coords.shape[0], nt=self.nt)
        self.rec.coordinates.data[:, :] = self.rec_coords

        pde = self.m * self.u.dt2 - self.u.laplace
        stencil = Eq(self.u.forward, solve(pde, self.u.forward))
        src_term = self.src.inject(field=self.u.forward, expr=self.src * (self.dt**2) / self.m)
        rec_term = self.rec.interpolate(expr=self.u.forward)
        self.operator = Operator([stencil] + src_term + rec_term, name="wavesleuth_acoustic_forward")

    def run(self, velocity_model: np.ndarray) -> SimulationResult:
        """Run the forward model for `velocity_model`."""
        vm = np.asarray(velocity_model, dtype=np.float32)
        if vm.shape != self.velocity_shape:
            raise ValidationError(f"Velocity model shape {vm.shape} does not match world grid {self.velocity_shape}.")
        if not np.all(np.isfinite(vm)) or float(np.min(vm)) <= 0.0:
            raise ValidationError("Velocity model must contain finite positive velocities.")
        validate_stability(self.world, vm)

        self.m.data[:, :] = (1.0 / (vm * vm)).astype(np.float32)
        self.u.data[:] = 0.0
        self.rec.data[:] = 0.0
        self.operator(time=self.nt - 2, dt=self.dt)

        traces = np.asarray(self.rec.data, dtype=np.float32).copy()
        final_wavefield: np.ndarray | None = None
        snapshots: np.ndarray | None = None
        if self.save_wavefield:
            full = np.asarray(self.u.data, dtype=np.float32)
            final_wavefield = full[-1].copy()
            stride = max(1, self.nt // 12)
            snapshots = full[::stride].copy()
        return SimulationResult(
            receiver_traces=traces,
            time=self.time.copy(),
            velocity_model=vm.copy(),
            source_coordinates=self.src_coords.copy(),
            receiver_coordinates=self.rec_coords.copy(),
            final_wavefield=final_wavefield,
            snapshots=snapshots,
            world=json.loads(json.dumps(self.world)),
        )


def simulate_world(
    world: dict[str, Any],
    *,
    out_path: str | None = None,
    save_wavefield: bool = True,
    quiet: bool = False,
) -> SimulationResult:
    """Create the velocity model, run Devito, and optionally save a `.npz` run."""
    validate_world(world)
    velocity_model = velocity_model_from_world(world)
    validate_stability(world, velocity_model)
    solver = DevitoAcoustic2D(world, save_wavefield=save_wavefield, quiet=quiet)
    result = solver.run(velocity_model)
    if out_path is not None:
        save_simulation_result(result, out_path)
    return result


def save_simulation_result(result: SimulationResult, path: str) -> None:
    """Save a `SimulationResult` to disk."""
    save_run_npz(
        path,
        receiver_traces=result.receiver_traces,
        time=result.time,
        velocity_model=result.velocity_model,
        source_coordinates=result.source_coordinates,
        receiver_coordinates=result.receiver_coordinates,
        final_wavefield=result.final_wavefield,
        snapshots=result.snapshots,
        world_json=json.dumps(result.world, sort_keys=True),
    )
