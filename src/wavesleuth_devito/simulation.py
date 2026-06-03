"""Devito-backed acoustic forward simulation."""

from __future__ import annotations

import copy
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
    shot_mode: str = "simultaneous"


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

    def run_for_source(self, velocity_model: np.ndarray, source_coordinate: np.ndarray | list[float] | tuple[float, float]) -> "SimulationResult":
        """Run a single-source shot by updating the sparse source coordinate."""
        if self.src_coords.shape[0] != 1:
            raise ValidationError("run_for_source requires a solver constructed with exactly one source point.")
        coord = np.asarray(source_coordinate, dtype=np.float32).reshape(1, 2)
        self.src.coordinates.data[:, :] = coord
        old = self.src_coords.copy()
        self.src_coords = coord.copy()
        try:
            return self.run(velocity_model)
        finally:
            self.src_coords = old

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


def _single_source_world(world: dict[str, Any], source_coordinate: np.ndarray) -> dict[str, Any]:
    single = copy.deepcopy(world)
    single["acquisition"]["sources"] = [{"x": float(source_coordinate[0]), "z": float(source_coordinate[1])}]
    return single



class ForwardTraceEngine:
    """Reusable forward trace engine for repeated candidate simulations.

    Inversion evaluates many velocity models with identical grid, time axis,
    sources, and receivers. Reusing the same Devito operator matters a lot:
    compiling a new operator for every candidate is much slower than updating the
    model parameter and rerunning the compiled operator.
    """

    def __init__(
        self,
        world: dict[str, Any],
        *,
        shot_mode: str | None = None,
        save_wavefield: bool = False,
        quiet: bool = False,
    ) -> None:
        validate_world(world)
        self.world = json.loads(json.dumps(world))
        self.mode = shot_mode or str(self.world.get("simulation", {}).get("shot_mode", "simultaneous"))
        if self.mode not in {"simultaneous", "sequential"}:
            raise ValidationError("shot_mode must be 'simultaneous' or 'sequential'.")
        self.world.setdefault("simulation", {})["shot_mode"] = self.mode
        self.src_coords = source_coordinates(self.world)
        self.rec_coords = receiver_coordinates(self.world)
        self.nt = int(self.world["simulation"]["nt"])
        self.dt = float(self.world["simulation"]["dt"])
        self.time = (np.arange(self.nt, dtype=np.float32) * self.dt).astype(np.float32)
        self.save_wavefield = bool(save_wavefield)
        if self.mode == "simultaneous" or self.src_coords.shape[0] == 1:
            self.solver = DevitoAcoustic2D(self.world, save_wavefield=save_wavefield, quiet=quiet)
            self._sequential = False
        else:
            single_world = _single_source_world(self.world, self.src_coords[0])
            self.solver = DevitoAcoustic2D(single_world, save_wavefield=save_wavefield, quiet=quiet)
            self._sequential = True

    def run(self, velocity_model: np.ndarray) -> SimulationResult:
        """Run the engine for one velocity model."""
        vm = np.asarray(velocity_model, dtype=np.float32)
        if not self._sequential:
            result = self.solver.run(vm)
            result.source_coordinates = self.src_coords.copy()
            result.receiver_coordinates = self.rec_coords.copy()
            result.world = json.loads(json.dumps(self.world))
            result.shot_mode = self.mode
            return result

        traces: list[np.ndarray] = []
        final_wavefield: np.ndarray | None = None
        snapshots: np.ndarray | None = None
        for coord in self.src_coords:
            result = self.solver.run_for_source(vm, coord)
            traces.append(result.receiver_traces)
            final_wavefield = result.final_wavefield
            snapshots = result.snapshots
        return SimulationResult(
            receiver_traces=np.stack(traces, axis=0).astype(np.float32),
            time=self.time.copy(),
            velocity_model=vm.copy(),
            source_coordinates=self.src_coords.copy(),
            receiver_coordinates=self.rec_coords.copy(),
            final_wavefield=final_wavefield,
            snapshots=snapshots,
            world=json.loads(json.dumps(self.world)),
            shot_mode=self.mode,
        )

def simulate_traces_for_velocity_model(
    world: dict[str, Any],
    velocity_model: np.ndarray,
    *,
    shot_mode: str | None = None,
    quiet: bool = False,
) -> np.ndarray:
    """Return receiver traces for a supplied velocity model.

    `shot_mode='simultaneous'` fires all sources at once and returns `(nt, nrec)`.
    `shot_mode='sequential'` fires each source separately and returns
    `(nshot, nt, nrec)` when there is more than one source. With one source, it
    returns `(nt, nrec)` for backward compatibility with the original MVP.
    """
    mode = shot_mode or str(world.get("simulation", {}).get("shot_mode", "simultaneous"))
    if mode not in {"simultaneous", "sequential"}:
        raise ValidationError("shot_mode must be 'simultaneous' or 'sequential'.")
    src_coords = source_coordinates(world)
    if mode == "simultaneous" or src_coords.shape[0] == 1:
        solver = DevitoAcoustic2D(world, save_wavefield=False, quiet=quiet)
        return solver.run(velocity_model).receiver_traces

    single_world = _single_source_world(world, src_coords[0])
    solver = DevitoAcoustic2D(single_world, save_wavefield=False, quiet=quiet)
    traces = [solver.run_for_source(velocity_model, coord).receiver_traces for coord in src_coords]
    return np.stack(traces, axis=0).astype(np.float32)


def simulate_velocity_model(
    world: dict[str, Any],
    velocity_model: np.ndarray,
    *,
    shot_mode: str | None = None,
    save_wavefield: bool = True,
    quiet: bool = False,
) -> SimulationResult:
    """Run a supplied velocity model and return a complete simulation result."""
    validate_world(world)
    vm = np.asarray(velocity_model, dtype=np.float32)
    validate_stability(world, vm)
    mode = shot_mode or str(world.get("simulation", {}).get("shot_mode", "simultaneous"))
    if mode not in {"simultaneous", "sequential"}:
        raise ValidationError("shot_mode must be 'simultaneous' or 'sequential'.")
    world_for_metadata = json.loads(json.dumps(world))
    world_for_metadata.setdefault("simulation", {})["shot_mode"] = mode
    src_coords = source_coordinates(world)
    rec_coords = receiver_coordinates(world)
    nt = int(world["simulation"]["nt"])
    dt = float(world["simulation"]["dt"])
    time = (np.arange(nt, dtype=np.float32) * dt).astype(np.float32)

    if mode == "simultaneous" or src_coords.shape[0] == 1:
        solver = DevitoAcoustic2D(world_for_metadata, save_wavefield=save_wavefield, quiet=quiet)
        result = solver.run(vm)
        result.source_coordinates = src_coords.copy()
        result.shot_mode = mode
        result.world = world_for_metadata
        return result

    single_world = _single_source_world(world_for_metadata, src_coords[0])
    solver = DevitoAcoustic2D(single_world, save_wavefield=save_wavefield, quiet=quiet)
    traces: list[np.ndarray] = []
    final_wavefield: np.ndarray | None = None
    snapshots: np.ndarray | None = None
    for coord in src_coords:
        result = solver.run_for_source(vm, coord)
        traces.append(result.receiver_traces)
        final_wavefield = result.final_wavefield
        snapshots = result.snapshots
    return SimulationResult(
        receiver_traces=np.stack(traces, axis=0).astype(np.float32),
        time=time,
        velocity_model=vm.copy(),
        source_coordinates=src_coords.copy(),
        receiver_coordinates=rec_coords.copy(),
        final_wavefield=final_wavefield,
        snapshots=snapshots,
        world=world_for_metadata,
        shot_mode=mode,
    )


def simulate_world(
    world: dict[str, Any],
    *,
    out_path: str | None = None,
    save_wavefield: bool = True,
    quiet: bool = False,
    shot_mode: str | None = None,
) -> SimulationResult:
    """Create the velocity model, run Devito, and optionally save a `.npz` run."""
    validate_world(world)
    velocity_model = velocity_model_from_world(world)
    result = simulate_velocity_model(
        world,
        velocity_model,
        shot_mode=shot_mode,
        save_wavefield=save_wavefield,
        quiet=quiet,
    )
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
        shot_mode=result.shot_mode,
    )
