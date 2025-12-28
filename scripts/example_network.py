import torch

import sim2d


class SimulatorPureGNNExample(sim2d.Simulator):
    def __init__(self, sim_time, init_gnn_path, log_conf):
        super().__init__(sim_time, 0, dt=0.01, init_gnn_path=init_gnn_path, logging_config=log_conf)

    def build_model(self):
        floor = sim2d.Floor(0.0, 0.0)
        circle_1 = sim2d.Circle(torch.tensor([0.1, 1.0]), torch.tensor([0.0, 0.0]), 1.0, 0.0, 0.2)
        circle_2 = sim2d.Circle(torch.tensor([0.0, 0.5]), torch.tensor([0.0, 0.0]), 1.0, 0.0, 0.2)
        self.floor = floor
        self.shapes = [circle_1, circle_2]


log_conf = sim2d.LoggingConfig(True, True, True, "data/log_gnn.h5")
sim = SimulatorPureGNNExample(1.0, "data/gnn_datasets/model.pt", log_conf)
sim.run()
