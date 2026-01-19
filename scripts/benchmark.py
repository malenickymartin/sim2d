import argparse
from copy import deepcopy

import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm
import h5py

import sim2d
from sim2d.logger import EngineLogger


class SimulatorBenchmark(sim2d.Simulator):
    def __init__(self, pass_path, init_gnn_path):
        self.pass_path = pass_path
        self.pass_step = 0
        super().__init__(0, init_gnn_path=init_gnn_path)

    def build_model(self):
        with h5py.File(self.pass_path, "r") as f:
            config = f["init_config"]

            self.dt = config["dt"][()]
            self.gravity = torch.tensor(
                config["gravity"][()], dtype=torch.float32, device=self.device
            )
            self.newton_iters = int(config["newton_iters"][()])
            self.num_steps = len([k for k in f.keys() if k.startswith("step_")]) - 1

            self.shapes = []
            num_shapes = int(config["shapes"]["num_shapes"][()])
            masses = config["shapes"]["masses"][()]
            restitutions = config["shapes"]["restitutions"][()]
            radii = config["shapes"]["radii"][()]
            types = config["shapes"]["shape_types"][()]
            for i in range(num_shapes):
                if sim2d.shapes.int_to_shape(types[i]) == sim2d.shapes.Circle:
                    shape = sim2d.shapes.Circle(
                        translation=torch.zeros(2),
                        velocity=torch.zeros(2),
                        mass=float(masses[i]),
                        restitution=float(restitutions[i]),
                        radius=radii[i],
                    )
                elif sim2d.shapes.int_to_shape(types[i]) == sim2d.shapes.Point:
                    shape = sim2d.shapes.Point(
                        translation=torch.zeros(2),
                        velocity=torch.zeros(2),
                        mass=float(masses[i]),
                        restitution=float(restitutions[i]),
                    )
                else:
                    raise NotImplementedError("Unknown shape type")
                self.shapes.append(shape)

            if config["floor"]["active"][()]:
                self.floor = sim2d.shapes.Floor(
                    height=float(config["floor"]["height"][()]),
                    restitution=float(config["floor"]["restitution"][()]),
                )

    def update(self):
        with h5py.File(self.pass_path, "r") as f:
            step_key = f"step_{self.pass_step:04d}"
            assert step_key in f, f"step_{self.pass_step:04d} not in hdf5"
            data = f[step_key]["shapes_data"]

            tran = torch.tensor(data["translation"][()], dtype=torch.float32)
            rot = torch.tensor(data["rotation"][()], dtype=torch.float32)
            vel = torch.tensor(data["velocity"][()], dtype=torch.float32)
            ang_vel = torch.tensor(data["angular_velocity"][()], dtype=torch.float32)

            for i, shape in enumerate(self.shapes):
                shape.translation = tran[i]
                shape.rotation = rot[i]
                shape.velocity = vel[i]
                shape.angular_velocity = ang_vel[i]
        self.pass_step += 1


def benchmark(dataset_root: str, model_name: str):
    test_dataset = dataset_root / "test_dataset" / "raw"
    model_path = dataset_root / "models" / model_name

    log_conf = sim2d.LoggingConfig(True, False, False, None)
    log_gnn = EngineLogger(log_conf)
    log_newton = EngineLogger(log_conf)
    log_hybrid = EngineLogger(log_conf)

    res = {"newton": [], "gnn": [], "hybrid": []}
    for pass_path in tqdm(test_dataset.iterdir()):
        Sim = SimulatorBenchmark(pass_path, model_path)
        for _ in range(Sim.num_steps):
            Sim.update()
            contacts, _ = Sim.collide()
            state = torch.zeros((Sim.num_shapes, 3), device=Sim.device)
            for j in range(Sim.num_shapes):
                state[j, :] = torch.cat(
                    [Sim.shapes[j].velocity, torch.tensor([Sim.shapes[j].angular_velocity])]
                )

            SimHybrid = deepcopy(Sim)
            SimHybrid.solver.logger = log_hybrid

            SimNewton = deepcopy(Sim)
            SimNewton.solver.logger = log_newton
            SimNewton.gnn = None

            SimGNN = deepcopy(Sim)
            SimGNN.solver.logger = log_gnn
            SimGNN.solver.newton_iters = 0

            state_hybrid = SimHybrid.solver.step(0, state, contacts)
            res_hybrid = torch.norm(Sim.solver.resudial_fn(state_hybrid, state, contacts)).item()

            state_newton = SimNewton.solver.step(0, state, contacts)
            res_newton = torch.norm(Sim.solver.resudial_fn(state_newton, state, contacts)).item()

            state_gnn = SimGNN.solver.step(0, state, contacts)
            res_gnn = torch.norm(Sim.solver.resudial_fn(state_gnn, state, contacts)).item()

            res["hybrid"].append(res_hybrid)
            res["newton"].append(res_newton)
            res["gnn"].append(res_gnn)

    print("===== RESIDUE =====")
    print(f"Hybrid: {np.median(res["hybrid"])*1e7:.4f}/1e7")
    print(f"Newton: {np.median(res["newton"])*1e7:.4f}/1e7")
    print(f"GNN: {np.median(res["gnn"])*1e7:.4f}/1e7")

    print("\nHybrid:")
    log_hybrid.print_timings()
    print("\nNewton:")
    log_newton.print_timings()
    print("\nGNN:")
    log_gnn.print_timings()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark")
    parser.add_argument("--dataset_root", type=Path)
    parser.add_argument("--model_name", type=str)
    args = parser.parse_args()
    benchmark(args.dataset_root, args.model_name)
