import torch

import sim2d
from sim2d.joints import RevoluteJoint

GNN_MODEL = "data/models/model_res_loss.pt"

PENDULUM_CFG = [
    [[[-1.0, 5.0], -torch.pi / 2], [[-3.0, 5.0], -torch.pi / 2]],
    [[[-1.0, 5.0], -torch.pi / 2], [[-2.0, 6.0], -torch.pi]],
    [[[0.0, 4.0], 0.0], [[-1.0, 3.0], -torch.pi / 2]],
]


class DoublePendulumSim(sim2d.Simulator):
    def __init__(self, sim_time, log_conf, run_idx, gnn_path, newton_iters):
        self.run_idx = run_idx
        super().__init__(
            sim_time,
            newton_iters=newton_iters,
            dt=0.01,
            logging_config=log_conf,
            init_gnn_path=gnn_path,
        )

    def build_model(self):
        link1 = sim2d.Rectangle(
            translation=PENDULUM_CFG[self.run_idx][0][0],
            rotation=PENDULUM_CFG[self.run_idx][0][1],
            velocity=torch.tensor([0.0, 0.0]),
            angular_velocity=0.0,
            mass=10.0,
            restitution=0.0,
            sides=[0.2, 2.0],
        )
        link2 = sim2d.Rectangle(
            translation=PENDULUM_CFG[self.run_idx][1][0],
            rotation=PENDULUM_CFG[self.run_idx][1][1],
            velocity=torch.tensor([0.0, 0.0]),
            angular_velocity=0.0,
            mass=10.0,
            restitution=0.0,
            sides=[0.2, 2.0],
        )
        self.shapes = [link1, link2]
        self.ignore_contacts = [(0, 1)]
        self.floor = sim2d.Floor(0.0, 0.0)
        self.joints = [
            RevoluteJoint(
                child_idx=0,
                parent_idx=-1,
                child_anchor=[0.0, 1.0],
                parent_anchor=[0.0, 5.0],
            ),
            RevoluteJoint(
                child_idx=1,
                parent_idx=0,
                child_anchor=[0.0, 1.0],
                parent_anchor=[0.0, -1.0],
            ),
        ]


def run(run_idx, gnn_path, newton_iters, tag):
    out_file = f"data/double_pendulum_{tag}_{run_idx}_res.h5"
    log_conf = sim2d.LoggingConfig(True, True, True, str(out_file))
    sim = DoublePendulumSim(5.0, log_conf, run_idx, gnn_path, newton_iters)
    sim.run()
    print(f"saved → {out_file}")


if __name__ == "__main__":
    for i in range(3):
        run(i, None, 100, "sim2d")
        run(i, GNN_MODEL, 0, "gnn")
