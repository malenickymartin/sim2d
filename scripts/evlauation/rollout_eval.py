import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

import sim2d
from sim2d.joints import FixedJoint, PrismaticJoint, RevoluteJoint


class RolloutSimulator(sim2d.Simulator):
    def __init__(self, pass_path: Path, device: torch.device):
        self._pass_path = pass_path
        self._device_ref = device
        super().__init__(sim_time=0, device=device)

    def build_model(self):
        with h5py.File(self._pass_path, "r") as f:
            cfg = f["init_config"]

            self.dt = float(cfg["dt"][()])
            self.gravity = torch.tensor(
                cfg["gravity"][()], dtype=torch.float32, device=self._device_ref
            )
            self.newton_iters = 0

            # Shapes
            num_shapes = int(cfg["shapes"]["num_shapes"][()])
            masses = cfg["shapes"]["masses"][()]
            restitutions = cfg["shapes"]["restitutions"][()]
            radii = cfg["shapes"]["radii"][()]
            types = cfg["shapes"]["shape_types"][()]

            self.shapes = []
            for i in range(num_shapes):
                shape_cls = sim2d.shapes.int_to_shape(types[i])
                kwargs = dict(
                    translation=torch.zeros(2),
                    rotation=torch.tensor(0.0),
                    velocity=torch.zeros(2),
                    angular_velocity=torch.tensor(0.0),
                    mass=float(masses[i]),
                    restitution=float(restitutions[i]),
                )
                if shape_cls == sim2d.shapes.Circle:
                    self.shapes.append(sim2d.shapes.Circle(**kwargs, radius=float(radii[i])))
                elif shape_cls == sim2d.shapes.Point:
                    self.shapes.append(sim2d.shapes.Point(**kwargs))
                else:
                    raise NotImplementedError(f"Unsupported shape type: {types[i]}")

            ic = cfg["shapes"]["ignored_contacts"][()]
            self.ignore_contacts = [tuple(pair) for pair in ic] if ic.ndim == 2 else []

            if cfg["floor"]["active"][()]:
                self.floor = sim2d.shapes.Floor(
                    height=float(cfg["floor"]["height"][()]),
                    restitution=float(cfg["floor"]["restitution"][()]),
                )

            num_joints = int(cfg["joints"]["num_joints"][()])
            self.joints = []
            if num_joints > 0:
                joint_types = cfg["joints"]["joint_types"][()]
                child_idxs = cfg["joints"]["child_idxs"][()]
                parent_idxs = cfg["joints"]["parent_idxs"][()]
                child_anchors = cfg["joints"]["child_anchors"][()]
                parent_anchors = cfg["joints"]["parent_anchors"][()]
                child_target_rots = cfg["joints"]["child_target_rotation"][()]
                parent_target_rots = cfg["joints"]["parent_target_rotation"][()]
                axes = cfg["joints"]["axis"][()]

                for i in range(num_joints):
                    jtype = int(joint_types[i])
                    child_idx = int(child_idxs[i])
                    parent_idx = int(parent_idxs[i])
                    ca = torch.tensor(child_anchors[i], dtype=torch.float32)
                    pa = torch.tensor(parent_anchors[i], dtype=torch.float32)

                    if jtype == 0:
                        joint = FixedJoint(
                            child_idx,
                            parent_idx,
                            ca,
                            pa,
                            float(child_target_rots[i]),
                            float(parent_target_rots[i]),
                        )
                    elif jtype == 1:
                        joint = RevoluteJoint(child_idx, parent_idx, ca, pa)
                    elif jtype == 2:
                        joint = PrismaticJoint(
                            child_idx,
                            parent_idx,
                            ca,
                            pa,
                            torch.tensor(axes[i], dtype=torch.float32),
                        )
                    else:
                        raise NotImplementedError(f"Unsupported joint type: {jtype}")
                    self.joints.append(joint)


def _set_shapes(sim: RolloutSimulator, tran, rot, vel, ang_vel):
    for i, shape in enumerate(sim.shapes):
        shape.translation = tran[i].clone()
        shape.rotation = rot[i].clone()
        shape.velocity = vel[i].clone()
        shape.angular_velocity = ang_vel[i].clone()


def _angle_diff(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    diff = pred - gt
    return np.abs((diff + np.pi) % (2 * np.pi) - np.pi)


def run_rollout(
    sim: RolloutSimulator,
    gnn,
    num_steps: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    n = sim.num_shapes
    translations = np.zeros((num_steps + 1, n, 2))
    rotations = np.zeros((num_steps + 1, n))
    velocities = np.zeros((num_steps + 1, n, 2))
    angular_velocities = np.zeros((num_steps + 1, n))

    for i, shape in enumerate(sim.shapes):
        translations[0, i] = shape.translation.cpu().numpy()
        rotations[0, i] = shape.rotation.cpu().numpy()
        velocities[0, i] = shape.velocity.cpu().numpy()
        angular_velocities[0, i] = shape.angular_velocity.cpu().numpy()

    with torch.no_grad():
        for step in range(num_steps):
            state = torch.zeros((n, 3), device=device)
            for i, shape in enumerate(sim.shapes):
                state[i, :2] = shape.velocity
                state[i, 2] = shape.angular_velocity

            try:
                contacts, _ = sim.collide()
                joints, _ = sim.process_joints()
                constraints = sim.merge_constraints(contacts, joints)

                gnn_data = sim.create_gnn_data(state, constraints)
                object_states, _ = gnn(
                    gnn_data.x_dict, gnn_data.edge_index_dict, gnn_data.edge_attr_dict
                )

                sim.update_shapes(object_states)
            except (AssertionError, RuntimeError):
                translations[step + 1 :] = np.nan
                rotations[step + 1 :] = np.nan
                velocities[step + 1 :] = np.nan
                angular_velocities[step + 1 :] = np.nan
                break

            for i, shape in enumerate(sim.shapes):
                translations[step + 1, i] = shape.translation.cpu().numpy()
                rotations[step + 1, i] = shape.rotation.cpu().numpy()
                velocities[step + 1, i] = shape.velocity.cpu().numpy()
                angular_velocities[step + 1, i] = shape.angular_velocity.cpu().numpy()

    return translations, rotations, velocities, angular_velocities


def main():
    parser = argparse.ArgumentParser(description="Evaluate GNN rollout accuracy")
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=Path("data/longer_runs/dataset_non_articulated/gnn_res"),
        help="Directory to save/load per-pass prediction caches (.npz). Skips rollout if cache exists.",
    )
    parser.add_argument(
        "--stats_cache",
        type=Path,
        default=Path("data/longer_runs/dataset_non_articulated/stats_res_new.npz"),
        help="Path to save/load aggregated per-step stats (.npz). Skips all rollout processing if it exists.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.stats_cache is not None and args.stats_cache.exists():
        print(f"Loading stats from {args.stats_cache}")
        sc = np.load(args.stats_cache)
        mean_trans = sc["trans_median"], sc["trans_p25"], sc["trans_p75"]
        mean_rot = sc["rot_median"], sc["rot_p25"], sc["rot_p75"]
        mean_vel = sc["vel_median"], sc["vel_p25"], sc["vel_p75"]
        mean_ang_vel = sc["ang_vel_median"], sc["ang_vel_p25"], sc["ang_vel_p75"]
        max_steps = len(mean_trans[0])
    else:
        if args.cache_dir is not None:
            args.cache_dir.mkdir(parents=True, exist_ok=True)

        raw_dir = args.dataset_root / "raw"
        pass_paths = sorted(raw_dir.glob("pass_*.h5"), key=lambda p: int(p.stem.split("_")[1]))
        print(f"Found {len(pass_paths)} passes in {raw_dir}")

        needs_gnn = args.cache_dir is None or any(
            not (args.cache_dir / f"{p.stem}.npz").exists() for p in pass_paths
        )
        if needs_gnn:
            gnn = torch.load(args.model_path, map_location=device, weights_only=False)
            gnn.eval()
        else:
            gnn = None
            print("All passes cached — skipping model load.")

        trans_pred_per_run = []
        trans_gt_per_run = []
        rot_pred_per_run = []
        rot_gt_per_run = []
        vel_pred_per_run = []
        vel_gt_per_run = []
        ang_vel_pred_per_run = []
        ang_vel_gt_per_run = []

        for pass_path in tqdm(pass_paths):
            with h5py.File(pass_path, "r") as f:
                step_keys = sorted(k for k in f.keys() if k.startswith("step_"))
                num_gt_steps = len(step_keys)
                num_shapes = int(f["init_config"]["shapes"]["num_shapes"][()])
                num_steps = min(args.num_steps, num_gt_steps - 1)

                gt_tran = np.zeros((num_steps + 1, num_shapes, 2))
                gt_rot = np.zeros((num_steps + 1, num_shapes))
                gt_vel = np.zeros((num_steps + 1, num_shapes, 2))
                gt_ang_vel = np.zeros((num_steps + 1, num_shapes))
                for s in range(num_steps + 1):
                    sd = f[f"step_{s:04d}"]["shapes_data"]
                    gt_tran[s] = sd["translation"][()]
                    gt_rot[s] = sd["rotation"][()]
                    gt_vel[s] = sd["velocity"][()]
                    gt_ang_vel[s] = sd["angular_velocity"][()]

                s0 = f["step_0000"]["shapes_data"]
                init_tran = torch.tensor(s0["translation"][()], dtype=torch.float32, device=device)
                init_rot = torch.tensor(s0["rotation"][()], dtype=torch.float32, device=device)
                init_vel = torch.tensor(s0["velocity"][()], dtype=torch.float32, device=device)
                init_ang = torch.tensor(
                    s0["angular_velocity"][()], dtype=torch.float32, device=device
                )

            cache_file = (
                args.cache_dir / f"{pass_path.stem}.npz" if args.cache_dir is not None else None
            )
            loaded_from_cache = cache_file is not None and cache_file.exists()

            if loaded_from_cache:
                cached = np.load(cache_file)
                pred_tran = cached["translations"]
                pred_rot = cached["rotations"]
                pred_vel = cached["velocities"]
                pred_ang_vel = cached["angular_velocities"]
            else:
                sim = RolloutSimulator(pass_path, device)
                _set_shapes(sim, init_tran, init_rot, init_vel, init_ang)
                pred_tran, pred_rot, pred_vel, pred_ang_vel = run_rollout(
                    sim, gnn, num_steps, device
                )

            # Slice to num_steps in case cache was saved with more steps
            pred_tran = pred_tran[: num_steps + 1]
            pred_rot = pred_rot[: num_steps + 1]
            pred_vel = pred_vel[: num_steps + 1]
            pred_ang_vel = pred_ang_vel[: num_steps + 1]

            # Per-object errors: shape (num_steps, num_shapes)
            trans_err = np.linalg.norm(pred_tran[1:] - gt_tran[1:], axis=-1)
            rot_err = _angle_diff(pred_rot[1:], gt_rot[1:])
            vel_err = np.linalg.norm(pred_vel[1:] - gt_vel[1:], axis=-1)
            ang_vel_err = np.abs(pred_ang_vel[1:] - gt_ang_vel[1:])

            if not loaded_from_cache and cache_file is not None:
                np.savez(
                    cache_file,
                    translations=pred_tran,
                    rotations=pred_rot,
                    velocities=pred_vel,
                    angular_velocities=pred_ang_vel,
                    trans_err=trans_err,
                    rot_err=rot_err,
                    vel_err=vel_err,
                    ang_vel_err=ang_vel_err,
                )

            trans_pred_per_run.append(pred_tran)
            trans_gt_per_run.append(gt_tran)
            rot_pred_per_run.append(pred_rot)
            rot_gt_per_run.append(gt_rot)
            vel_pred_per_run.append(pred_vel)
            vel_gt_per_run.append(gt_vel)
            ang_vel_pred_per_run.append(pred_ang_vel)
            ang_vel_gt_per_run.append(gt_ang_vel)

        max_steps = max(len(t) for t in trans_gt_per_run)

        def per_step_stats(predictions, ground_truth, distance_normalization=False):

            normalization = [
                [[1.0 for _ in range(len(ground_truth[i][j]))] for j in range(len(ground_truth[i]))]
                for i in range(len(ground_truth))
            ]
            for run in range(len(ground_truth)):
                for step in range(1, len(ground_truth[run])):
                    for obj in range(len(ground_truth[run][step])):
                        if distance_normalization:
                            dist = np.linalg.norm(
                                ground_truth[run][step][obj] - ground_truth[run][step - 1][obj]
                            )
                            normalization[run][step][obj] = normalization[run][step - 1][obj] + dist
                        else:
                            normalization[run][step][obj] = np.linalg.norm(
                                ground_truth[run][step][obj] - ground_truth[run][0][obj]
                            )

            errors_all = [[] for _ in range(len(ground_truth[0]))]
            for run in range(len(ground_truth)):
                for step in range(len(ground_truth[run])):
                    for obj in range(len(ground_truth[run][step])):
                        err = (
                            np.linalg.norm(
                                ground_truth[run][step][obj] - predictions[run][step][obj]
                            )
                            / normalization[run][step][obj]
                        )

                        errors_all[step].append(err)
            errors = np.zeros(len(ground_truth[0]))
            percentile_25 = np.zeros(len(ground_truth[0]))
            percentile_75 = np.zeros(len(ground_truth[0]))
            for step in range(len(ground_truth[run])):
                errors[step] = np.median(errors_all[step])
                percentile_25[step] = np.percentile(errors_all[step], 25)
                percentile_75[step] = np.percentile(errors_all[step], 75)
            return errors * 100, percentile_25 * 100, percentile_75 * 100

        mean_trans = per_step_stats(trans_pred_per_run, trans_gt_per_run)
        mean_rot = per_step_stats(rot_pred_per_run, rot_gt_per_run)
        mean_vel = per_step_stats(vel_pred_per_run, vel_gt_per_run)
        mean_ang_vel = per_step_stats(ang_vel_pred_per_run, ang_vel_gt_per_run)

        if args.stats_cache is not None:
            np.savez(
                args.stats_cache,
                trans_median=mean_trans[0],
                trans_p25=mean_trans[1],
                trans_p75=mean_trans[2],
                rot_median=mean_rot[0],
                rot_p25=mean_rot[1],
                rot_p75=mean_rot[2],
                vel_median=mean_vel[0],
                vel_p25=mean_vel[1],
                vel_p75=mean_vel[2],
                ang_vel_median=mean_ang_vel[0],
                ang_vel_p25=mean_ang_vel[1],
                ang_vel_p75=mean_ang_vel[2],
            )
            print(f"Saved stats to {args.stats_cache}")


if __name__ == "__main__":
    main()
