from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import Union

import h5py
import numpy as np
import pandas as pd
import torch

from .shapes import Shape
from .shapes import Floor
from .shapes import shape_to_int


@dataclass
class LoggingConfig:
    enable_timing: bool = True
    enable_hdf5: bool = False
    enable_detailed_hdf5: bool = False
    log_file: str = "simulation_log.h5"


class HDF5Logger:
    def __init__(self, filepath: str, mode: str = "w"):
        self.filepath = filepath
        self.mode = mode
        self._file = None
        self._scope_stack = [""]

    def open(self):
        if self._file is not None:
            return

        directory = os.path.dirname(self.filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)

        self._file = h5py.File(self.filepath, self.mode)
        print(f"HDF5Logger: Opened {self.filepath}")

    def close(self):
        if self._file is None:
            return
        self._file.close()
        self._file = None
        print(f"HDF5Logger: Closed {self.filepath}")

    @contextmanager
    def scope(self, name: str):
        self._scope_stack.append(name)
        yield
        self._scope_stack.pop()

    @property
    def current_path(self) -> str:
        return "/".join(s for s in self._scope_stack if s)

    def log_data(self, name: str, data: Any):
        if self._file is None:
            return

        group = self._file
        path = self.current_path
        if path:
            if path not in group:
                group = group.create_group(path)
            else:
                group = group[path]

        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        if name in group:
            del group[name]
        group.create_dataset(name, data=data)


class EngineLogger:
    def __init__(self, config: LoggingConfig):
        self.config = config
        self.hdf5_logger = HDF5Logger(config.log_file) if config.enable_hdf5 else None

        self.timings: Dict[str, list] = {}

    def open(self):
        if self.hdf5_logger:
            self.hdf5_logger.open()

    def close(self):
        if self.hdf5_logger:
            self.hdf5_logger.close()
        self.print_timings()

    @contextmanager
    def timed_block(self, name: str):
        if not self.config.enable_timing:
            yield
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start = time.perf_counter()
        yield

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()

        duration_ms = (end - start) * 1000.0

        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(duration_ms)

    def log_init_config(self, sim):
        if not self.hdf5_logger:
            return

        with self.hdf5_logger.scope("init_config"):
            self.hdf5_logger.log_data("num_shapes", len(sim.shapes))
            self.hdf5_logger.log_data("shape_types", [shape_to_int(s) for s in sim.shapes])
            self.hdf5_logger.log_data("masses", [s.mass for s in sim.shapes])
            self.hdf5_logger.log_data("restitutions", [s.restitution for s in sim.shapes])
            radii = []
            for s in sim.shapes:
                radii.append(getattr(s, "radius", 0.0))
            self.hdf5_logger.log_data("radii", radii)

            self.hdf5_logger.log_data("gravity", sim.gravity)
            self.hdf5_logger.log_data("dt", sim.dt)
            self.hdf5_logger.log_data("newton_iters", sim.newton_iters)
            with self.hdf5_logger.scope("floor"):
                if not sim.floor is None:
                    self.hdf5_logger.log_data("active", True)
                    self.hdf5_logger.log_data("restitution", sim.floor.restitution)
                    self.hdf5_logger.log_data("height", sim.floor.height)
                else:
                    self.hdf5_logger.log_data("active", False)

    def log_step_data(
        self,
        step: int,
        time_val: float,
        shapes: list[Shape],
        state: torch.Tensor,
        contact_log: dict,
    ):
        if not self.hdf5_logger:
            return

        with self.hdf5_logger.scope(f"step_{step:04d}"):
            self.hdf5_logger.log_data("time", time_val)
            with self.hdf5_logger.scope("shapes_data"):
                self.hdf5_logger.log_data("translation", [s.translation for s in shapes])
                self.hdf5_logger.log_data("rotation", [s.rotation for s in shapes])
                self.hdf5_logger.log_data("velocity", [s.velocity for s in shapes])
                self.hdf5_logger.log_data("angular_velocity", [s.angular_velocity for s in shapes])
            with self.hdf5_logger.scope("contacts_data"):
                self.hdf5_logger.log_data("count", contact_log["count"])
                self.hdf5_logger.log_data("lambdas", state[:, 3:])
                self.hdf5_logger.log_data("indices", contact_log["indices"])
                self.hdf5_logger.log_data("distances", contact_log["distances"])
                self.hdf5_logger.log_data("Js", contact_log["Js"])

    def log_engine_data(
        self,
        step: int,
        newton_step: int,
        shape: torch.Tensor,
        res: torch.Tensor,
        delta: torch.Tensor,
        J: torch.Tensor,
    ):
        if not self.hdf5_logger or not self.config.enable_detailed_hdf5:
            return

        with self.hdf5_logger.scope(f"step_{step:04d}"):
            with self.hdf5_logger.scope("engine_data"):
                with self.hdf5_logger.scope(f"newton_step_{newton_step:04d}"):
                    self.hdf5_logger.log_data("J", J)
                    self.hdf5_logger.log_data("J_cond", torch.linalg.cond(J))
                    self.hdf5_logger.log_data("res", res.reshape(shape))
                    self.hdf5_logger.log_data(
                        "delta", delta.reshape(shape) if len(delta) > 0 else delta
                    )

    def print_timings(self):
        if not self.timings:
            return

        print("\n=== TIMING PERFORMANCE REPORT ===")
        data = []
        for name, values in self.timings.items():
            arr = np.array(values)
            data.append(
                {
                    "Operation": name,
                    "Mean (ms)": f"{np.mean(arr):.3f}",
                    "Std (ms)": f"{np.std(arr):.3f}",
                    "Min (ms)": f"{np.min(arr):.3f}",
                    "Max (ms)": f"{np.max(arr):.3f}",
                    "Calls": len(arr),
                }
            )

        df = pd.DataFrame(data)
        print(df.to_string(index=False))
