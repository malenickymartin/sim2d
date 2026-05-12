import argparse
from copy import deepcopy
from shutil import rmtree

import torch
from pathlib import Path
import numpy as np
from tqdm import tqdm
import h5py
from matplotlib import pyplot as plt

import sim2d
from sim2d.logger import EngineLogger


class SimulatorBenchmark(sim2d.Simulator):
    def __init__(self, pass_path):
        self.pass_path = pass_path
        self.pass_step = 0
        super().__init__(0, device=torch.device("cpu"))

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
                        rotation=torch.tensor(0.0),
                        velocity=torch.zeros(2),
                        angular_velocity=torch.tensor(0.0),
                        mass=float(masses[i]),
                        restitution=float(restitutions[i]),
                        radius=radii[i],
                    )
                elif sim2d.shapes.int_to_shape(types[i]) == sim2d.shapes.Point:
                    shape = sim2d.shapes.Point(
                        translation=torch.zeros(2),
                        rotation=torch.tensor(0.0),
                        velocity=torch.zeros(2),
                        angular_velocity=torch.tensor(0.0),
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

            ignored = config["shapes"]["ignored_contacts"][()]
            if ignored.ndim == 2 and ignored.shape[0] > 0:
                self.ignore_contacts = [tuple(pair) for pair in ignored]

            joints_config = config["joints"]
            num_joints = int(joints_config["num_joints"][()])
            if num_joints > 0:
                joint_types = joints_config["joint_types"][()]
                child_idxs = joints_config["child_idxs"][()]
                parent_idxs = joints_config["parent_idxs"][()]
                child_anchors = joints_config["child_anchors"][()]
                parent_anchors = joints_config["parent_anchors"][()]
                child_target_rotations = joints_config["child_target_rotation"][()]
                parent_target_rotations = joints_config["parent_target_rotation"][()]
                axes = joints_config["axis"][()]
                for i in range(num_joints):
                    joint_class = sim2d.joints.int_to_joint(int(joint_types[i]))
                    common = dict(
                        child_idx=int(child_idxs[i]),
                        parent_idx=int(parent_idxs[i]),
                        child_anchor=torch.tensor(child_anchors[i], dtype=torch.float32),
                        parent_anchor=torch.tensor(parent_anchors[i], dtype=torch.float32),
                    )
                    if joint_class == sim2d.FixedJoint:
                        joint = sim2d.FixedJoint(
                            **common,
                            child_target_rotation=float(child_target_rotations[i]),
                            parent_target_rotation=float(parent_target_rotations[i]),
                        )
                    elif joint_class == sim2d.RevoluteJoint:
                        joint = sim2d.RevoluteJoint(**common)
                    elif joint_class == sim2d.PrismaticJoint:
                        joint = sim2d.PrismaticJoint(
                            **common,
                            axis=torch.tensor(axes[i], dtype=torch.float32),
                        )
                    else:
                        raise NotImplementedError(f"Unknown joint type: {joint_class}")
                    self.joints.append(joint)

    def update(self):
        with h5py.File(self.pass_path, "r") as f:
            step_key = f"step_{self.pass_step:04d}"
            assert step_key in f, f"step_{self.pass_step:04d} not in hdf5"
            data = f[step_key]["shapes_data"]

            tran = torch.tensor(data["translation"][()], dtype=torch.float32, device=self.device)
            rot = torch.tensor(data["rotation"][()], dtype=torch.float32, device=self.device)
            vel = torch.tensor(data["velocity"][()], dtype=torch.float32, device=self.device)
            ang_vel = torch.tensor(
                data["angular_velocity"][()], dtype=torch.float32, device=self.device
            )

            for i, shape in enumerate(self.shapes):
                shape.translation = tran[i]
                shape.rotation = rot[i]
                shape.velocity = vel[i]
                shape.angular_velocity = ang_vel[i]
        self.pass_step += 1


def load_residues(filename):
    all_runs = []
    with h5py.File(filename, "r") as f:
        steps = sorted(f.keys(), key=lambda x: int(x) if x.isdigit() else x)
        for step in steps:
            run_residues = []
            engine_data = f[step]["engine_data"]
            newton_steps = sorted(engine_data.keys(), key=lambda x: int(x) if x.isdigit() else x)
            for ns in newton_steps:
                val = np.linalg.norm(engine_data[ns]["res"][()])
                run_residues.append(val)

            if run_residues:
                all_runs.append(run_residues)
    return all_runs


def pad_data_matrix(data_list):
    max_len = max(len(run) for run in data_list)
    padded_matrix = np.zeros((len(data_list), max_len))
    for i, run in enumerate(data_list):
        length = len(run)
        padded_matrix[i, :length] = run
        padded_matrix[i, length:] = run[-1]
    return padded_matrix


def plot_spaghetti(ax, data, label, color):
    for run in data:
        ax.plot(run, color=color, alpha=0.1, linewidth=1)
    ax.plot([], [], color=color, label=label)
    ax.set_title("Raw Data")
    ax.set_yscale("log")
    ax.set_xlabel("Newton Step")
    ax.set_ylabel("Residue Norm")
    ax.set_xlim([0, 7])
    ax.grid(True, which="both", linestyle="--", alpha=0.5)


def plot_extended_tail(ax, data, label, color):
    matrix = pad_data_matrix(data)
    median = np.median(matrix, axis=0)
    p25 = np.percentile(matrix, 25, axis=0)
    p75 = np.percentile(matrix, 75, axis=0)
    x = np.arange(len(median))
    ax.plot(x, median, color=color, linewidth=2, label=f"{label} (Median)")
    ax.fill_between(x, p25, p75, color=color, alpha=0.2, label=f"{label} IQR")
    ax.set_title("Aggregated (Median + IQR)")
    ax.set_yscale("log")
    ax.set_xlabel("Newton Step")
    ax.set_ylabel("Residue Norm")
    ax.set_xlim([0, 7])
    ax.grid(True, which="both", linestyle="--", alpha=0.5)


def plot_convergence_prob(ax, data, label, color, threshold=1e-6):
    matrix = pad_data_matrix(data)
    n_runs, max_steps = matrix.shape
    converged_matrix = matrix < threshold
    convergence_counts = np.sum(converged_matrix, axis=0)
    convergence_rate = (convergence_counts / n_runs) * 100
    ax.plot(convergence_rate, color=color, linewidth=2, label=label)
    ax.set_title(f"Convergence Probability (< {threshold})")
    ax.set_ylabel("% Converged")
    ax.set_xlabel("Newton Step")
    ax.set_ylim(0, 105)
    ax.set_xlim([0, 7])
    ax.grid(True)


def run_benchmark(dataset_root: Path, model_name: str, tmp_dir_name: str):
    test_dataset = dataset_root / "raw"
    model_path = Path("data/models") / model_name

    log_hybrid = EngineLogger(sim2d.LoggingConfig(False, False, True, f"{tmp_dir_name}/hybrid.h5"))
    log_newton = EngineLogger(sim2d.LoggingConfig(False, False, True, f"{tmp_dir_name}/newton.h5"))
    log_hybrid.open()
    log_newton.open()

    gnn = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
    gnn.eval()

    step_total = 0
    for pass_path in tqdm(test_dataset.iterdir()):
        Sim = SimulatorBenchmark(pass_path)
        for _ in range(Sim.num_steps):
            Sim.update()
            contacts, _ = Sim.collide()
            joints, _ = Sim.process_joints()
            constraints = Sim.merge_constraints(contacts, joints)
            state = torch.zeros((Sim.num_shapes, 3), device=Sim.device)
            for j in range(Sim.num_shapes):
                state[j, :] = torch.cat(
                    [Sim.shapes[j].velocity, torch.tensor([Sim.shapes[j].angular_velocity])]
                )

            SimHybrid = deepcopy(Sim)
            SimHybrid.solver.logger = log_hybrid
            SimHybrid.gnn = gnn

            SimNewton = deepcopy(Sim)
            SimNewton.solver.logger = log_newton

            SimHybrid.solver.step(step_total, state, constraints)
            SimNewton.solver.step(step_total, state, constraints)
            step_total += 1

    log_newton.close()
    log_hybrid.close()


def plot(tmp_dir_name, plot_file_name):
    hybrid_data = load_residues(f"{tmp_dir_name}/hybrid.h5")
    newton_data = load_residues(f"{tmp_dir_name}/newton.h5")

    fig, axes = plt.subplots(3, 1, figsize=(18, 13), constrained_layout=True)

    plot_spaghetti(axes[0], hybrid_data, "Hybrid", "tab:blue")
    plot_spaghetti(axes[0], newton_data, "Newton", "tab:orange")
    axes[0].legend()

    plot_extended_tail(axes[1], hybrid_data, "Hybrid", "tab:blue")
    plot_extended_tail(axes[1], newton_data, "Newton", "tab:orange")
    axes[1].legend()

    plot_convergence_prob(axes[2], hybrid_data, "Hybrid", "tab:blue", threshold=1e-6)
    plot_convergence_prob(axes[2], newton_data, "Newton", "tab:orange", threshold=1e-6)
    axes[2].legend()

    plt.suptitle("Comparison of Initialization Methods", fontsize=16)
    plt.savefig(plot_file_name)


def main(dataset_root, model_name, keep_runs, plot_only):
    tmp_dir_name = str(dataset_root / f"_tmp_{model_name.split(".")[0]}")
    plot_file_name = str(dataset_root / f"_benchmark_{model_name.split(".")[0]}")
    if not plot_only:
        run_benchmark(dataset_root, model_name, tmp_dir_name)
    plot(tmp_dir_name, plot_file_name)
    if not keep_runs:
        rmtree(tmp_dir_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark")
    parser.add_argument("--dataset_root", type=Path, default=Path("data/gnn_dataset"))
    parser.add_argument("--model_name", type=str, default="model.pt")
    parser.add_argument("--keep_runs", action="store_true", default=False)
    parser.add_argument("--plot_only", action="store_true", default=False)
    args = parser.parse_args()
    main(args.dataset_root, args.model_name, args.keep_runs, args.plot_only)
