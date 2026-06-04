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


def sponge_damping_model(world: dict[str, Any]) -> np.ndarray:
    """Return a simple quadratic sponge damping field for optional edge damping.

    This is not a full PML. It is a conservative MVP sponge that reduces some
    boundary reflections without pretending to be a production absorbing layer.
    """
    validate_world(world)
    nx, nz = grid_shape(world)
    sim = world.get("simulation", {})
    if str(sim.get("boundary", "none")) != "sponge":
        return np.zeros((nx, nz), dtype=np.float32)
    width = int(sim.get("sponge_width", 0))
    strength = float(sim.get("sponge_strength", 0.0))
    if width <= 0 or strength <= 0.0:
        return np.zeros((nx, nz), dtype=np.float32)
    width = min(width, max(1, min(nx, nz) // 2 - 1))
    ix = np.minimum(np.arange(nx), nx - 1 - np.arange(nx))[:, None]
    iz = np.minimum(np.arange(nz), nz - 1 - np.arange(nz))[None, :]
    dist = np.minimum(ix, iz).astype(np.float32)
    taper = np.clip((float(width) - dist) / float(width), 0.0, 1.0)
    return (strength * taper * taper).astype(np.float32)


def _shift_1d_with_zeros(trace: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return trace.copy()
    out = np.zeros_like(trace)
    if shift > 0:
        out[shift:] = trace[:-shift]
    else:
        out[:shift] = trace[-shift:]
    return out


def apply_trace_noise(
    traces: np.ndarray,
    *,
    dt: float,
    noise_level: float = 0.0,
    receiver_dropout: float = 0.0,
    amplitude_jitter: float = 0.0,
    time_jitter: float = 0.0,
    seed: int = 20260203,
) -> np.ndarray:
    """Apply deterministic synthetic observation imperfections to traces.

    `noise_level` is relative to global trace RMS. `receiver_dropout` zeros a
    fraction of receiver channels. `amplitude_jitter` applies multiplicative
    per-channel gains. `time_jitter` shifts each shot/receiver trace by a small
    integer number of samples derived from seconds.
    """
    arr = np.asarray(traces, dtype=np.float32).copy()
    if arr.ndim not in {2, 3}:
        raise ValidationError(f"Trace noise supports 2D or 3D trace arrays, got shape {arr.shape}.")
    if noise_level < 0.0 or receiver_dropout < 0.0 or amplitude_jitter < 0.0 or time_jitter < 0.0:
        raise ValidationError("Noise parameters must be non-negative.")
    if receiver_dropout >= 1.0:
        raise ValidationError("receiver_dropout must be less than 1.0.")
    rng = np.random.default_rng(int(seed))

    if amplitude_jitter > 0.0:
        if arr.ndim == 2:
            gains = rng.normal(loc=1.0, scale=float(amplitude_jitter), size=(1, arr.shape[1]))
        else:
            gains = rng.normal(loc=1.0, scale=float(amplitude_jitter), size=(arr.shape[0], 1, arr.shape[2]))
        arr = (arr * gains.astype(np.float32)).astype(np.float32)

    if time_jitter > 0.0:
        max_shift = int(round(float(time_jitter) / float(dt)))
        if max_shift > 0:
            if arr.ndim == 2:
                for irec in range(arr.shape[1]):
                    shift = int(rng.integers(-max_shift, max_shift + 1))
                    arr[:, irec] = _shift_1d_with_zeros(arr[:, irec], shift)
            else:
                for ishot in range(arr.shape[0]):
                    for irec in range(arr.shape[2]):
                        shift = int(rng.integers(-max_shift, max_shift + 1))
                        arr[ishot, :, irec] = _shift_1d_with_zeros(arr[ishot, :, irec], shift)

    if receiver_dropout > 0.0:
        nrec = arr.shape[-1]
        keep = rng.random(nrec) >= float(receiver_dropout)
        if not bool(np.any(keep)):
            keep[int(rng.integers(0, nrec))] = True
        if arr.ndim == 2:
            arr[:, ~keep] = 0.0
        else:
            arr[:, :, ~keep] = 0.0

    if noise_level > 0.0:
        rms = float(np.sqrt(np.mean(arr * arr))) if np.any(arr) else 1.0
        scale = float(noise_level) * max(rms, 1.0e-12)
        arr = (arr + rng.normal(0.0, scale, size=arr.shape).astype(np.float32)).astype(np.float32)
    return arr.astype(np.float32)


def noise_config_from_world(world: dict[str, Any]) -> dict[str, float | int]:
    """Return normalized noise settings from world metadata."""
    noise = world.get("simulation", {}).get("noise", {}) or {}
    return {
        "noise_level": float(noise.get("noise_level", 0.0)),
        "receiver_dropout": float(noise.get("receiver_dropout", 0.0)),
        "amplitude_jitter": float(noise.get("amplitude_jitter", 0.0)),
        "time_jitter": float(noise.get("time_jitter", 0.0)),
        "seed": int(noise.get("seed", 20260203)),
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
        self.damp = Function(name="damp", grid=self.grid, space_order=0)
        self.damp.data[:, :] = sponge_damping_model(self.world)
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

        pde = self.m * self.u.dt2 + self.damp * self.u.dt - self.u.laplace
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
    """Reusable forward trace engine for repeated candidate simulations."""

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
    """Return receiver traces for a supplied velocity model."""
    engine = ForwardTraceEngine(world, shot_mode=shot_mode, save_wavefield=False, quiet=quiet)
    return engine.run(velocity_model).receiver_traces


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
    engine = ForwardTraceEngine(world_for_metadata, shot_mode=mode, save_wavefield=save_wavefield, quiet=quiet)
    return engine.run(vm)


def _merge_noise_overrides(world: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(world)
    base = noise_config_from_world(merged)
    explicitly_requested = False
    for key, value in overrides.items():
        if value is not None:
            base[key] = value
            explicitly_requested = True
    if explicitly_requested or any(float(base[k]) > 0.0 for k in ("noise_level", "receiver_dropout", "amplitude_jitter", "time_jitter")):
        merged.setdefault("simulation", {})["noise"] = {
            "noise_level": float(base["noise_level"]),
            "receiver_dropout": float(base["receiver_dropout"]),
            "amplitude_jitter": float(base["amplitude_jitter"]),
            "time_jitter": float(base["time_jitter"]),
            "seed": int(base["seed"]),
        }
    return merged


def simulate_world(
    world: dict[str, Any],
    *,
    out_path: str | None = None,
    save_wavefield: bool = True,
    quiet: bool = False,
    shot_mode: str | None = None,
    noise_level: float | None = None,
    receiver_dropout: float | None = None,
    amplitude_jitter: float | None = None,
    time_jitter: float | None = None,
    noise_seed: int | None = None,
) -> SimulationResult:
    """Create the velocity model, run Devito, optionally add observation noise, and save a `.npz`."""
    world_for_run = _merge_noise_overrides(
        world,
        {
            "noise_level": noise_level,
            "receiver_dropout": receiver_dropout,
            "amplitude_jitter": amplitude_jitter,
            "time_jitter": time_jitter,
            "seed": noise_seed,
        },
    )
    validate_world(world_for_run)
    velocity_model = velocity_model_from_world(world_for_run)
    result = simulate_velocity_model(
        world_for_run,
        velocity_model,
        shot_mode=shot_mode,
        save_wavefield=save_wavefield,
        quiet=quiet,
    )
    noise = noise_config_from_world(world_for_run)
    if any(float(noise[k]) > 0.0 for k in ("noise_level", "receiver_dropout", "amplitude_jitter", "time_jitter")):
        result.receiver_traces = apply_trace_noise(
            result.receiver_traces,
            dt=float(result.time[1] - result.time[0]) if result.time.shape[0] > 1 else float(world_for_run["simulation"]["dt"]),
            noise_level=float(noise["noise_level"]),
            receiver_dropout=float(noise["receiver_dropout"]),
            amplitude_jitter=float(noise["amplitude_jitter"]),
            time_jitter=float(noise["time_jitter"]),
            seed=int(noise["seed"]),
        )
        result.world = world_for_run
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
