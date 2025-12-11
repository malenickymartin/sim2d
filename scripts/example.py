import torch

import sim2d


class SimulatorExample(sim2d.Simulator):
    def __init__(self, sim_time, newton_iters, log_conf):
        super().__init__(sim_time, newton_iters, dt=0.01, logging_config=log_conf)

    def build_model(self):
        floor = sim2d.Floor(0.0, 0.0)
        circle_1 = sim2d.Circle(torch.tensor([0.0, 2.1]), torch.tensor([0.0, 0.0]), 1.0, 0.0, 0.2)
        self.floor = floor
        self.shapes = [circle_1]

    def init_state_fn(self, state, contacts, dt):
        return super().init_state_fn(state, contacts, dt)


log_conf = sim2d.LoggingConfig(True, True, "data/log.h5")
sim = SimulatorExample(1.0, 300, log_conf)
sim.run()
