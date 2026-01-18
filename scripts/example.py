import torch
import sim2d
from sim2d.joints import RevoluteJoint, PrismaticJoint, FixedJoint


class JointExampleSim(sim2d.Simulator):
    def __init__(self, sim_time, log_conf):
        super().__init__(sim_time, newton_iters=50, dt=0.01, logging_config=log_conf)

    def build_model(self):
        pendulum = sim2d.Rectangle(
            translation=torch.tensor([3.0, 2.5]),
            rotation=torch.tensor(torch.pi / 2),
            velocity=torch.tensor([0.0, 0.0]),
            angular_velocity=torch.tensor(-torch.pi),
            mass=torch.tensor(2.0),
            restitution=torch.tensor(0.9),
            sides=torch.tensor([0.2, 2.0]),
        )

        ball = sim2d.Circle(
            translation=torch.tensor([1.8, 0.7]),
            velocity=torch.tensor([0.0, 0.0]),
            mass=10.0,
            restitution=0.9,
            radius=0.3,
        )

        self.floor = sim2d.Floor(0.0, 0.5)
        self.shapes = [pendulum, ball]

        rev_joint = RevoluteJoint(
            child_idx=0,
            parent_idx=-1,
            child_anchor=torch.tensor([0.0, 1.0]),
            parent_anchor=torch.tensor([2.0, 2.5]),
        )

        self.joints = [rev_joint]


# Run Simulation
log_conf = sim2d.LoggingConfig(True, True, True, "data/log_joints.h5")
sim = JointExampleSim(1.0, log_conf)
sim.run()
