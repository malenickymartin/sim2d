import torch

import sim2d

GNN_MODEL = "data/models/model_res_loss.pt"

BALLS_CFG = [
    ([1.0, 1.0], 0.50, 10.000737),
    ([1.5, 2.0], 0.30, 9.9978045),
    ([0.0, 2.5], 0.40, 4.0212386),
    ([2.5, 1.5], 0.25, 3.272492),
    ([-0.5, 3.5], 0.35, 4.4898595),
    ([2.0, 0.5], 0.20, 2.1781709065),
    ([-0.5, 1.0], 0.45, 9.542587685),
]


class BallsSim(sim2d.Simulator):
    def __init__(self, sim_time, log_conf, gnn_path, newton_iters):
        super().__init__(
            sim_time,
            newton_iters=newton_iters,
            dt=0.01,
            logging_config=log_conf,
            init_gnn_path=gnn_path,
        )

    def build_model(self):
        self.shapes = [
            sim2d.Circle(
                translation=torch.tensor(pos),
                rotation=0,
                velocity=torch.tensor([0.0, 0.0]),
                angular_velocity=0,
                mass=mass,
                restitution=0.0,
                radius=radius,
            )
            for pos, radius, mass in BALLS_CFG
        ]
        self.floor = sim2d.Floor(0.0, 0.0)
        self.joints = []


def run(gnn_path, newton_iters, out_file):
    log_conf = sim2d.LoggingConfig(False, True, False, str(out_file))
    sim = BallsSim(2.0, log_conf, gnn_path, newton_iters)
    sim.run()
    print(f"saved → {out_file}")


if __name__ == "__main__":
    run(None, 100, "data/balls_sim2d.h5")
    run(GNN_MODEL, 0, "data/balls_gnn.h5")
