# Sim2D

Sim2D is a PyTorch-based 2D rigid-body simulator with constraint-based contacts and joints.
It also includes optional Graph Neural Network (GNN) tooling for learning constraint forces and
state updates from logged simulation data.

## Features

- Constraint-based dynamics with implicit Euler integration and Newton iterations.
- Shapes: `Circle`, `Rectangle`, `Point`, and a static `Floor`.
- Contacts with restitution and joint constraints (fixed, revolute, prismatic).
- HDF5 logging for simulation states, contacts, joints, and timing diagnostics.
- PyTorch Geometric dataset + model utilities for training a GNN surrogate.

## Requirements

- Python 3.12+
- PyTorch and PyTorch Geometric (match versions to your CUDA/CPU setup)

The project dependencies are listed in `pyproject.toml`. Install PyTorch and
PyTorch Geometric following their official instructions if needed.

## Installation

From the repository root:

```bash
pip install -e .
```

## Quick start

Run the built-in joint example (writes `data/log_joints.h5`):

```bash
python scripts/example.py
```

Minimal custom simulation:

```python
import torch
import sim2d


class MySim(sim2d.Simulator):
    def __init__(self, sim_time, log_conf):
        super().__init__(sim_time, newton_iters=50, dt=0.01, logging_config=log_conf)

    def build_model(self):
        ball = sim2d.Circle(
            translation=torch.tensor([0.0, 1.0]),
            rotation=torch.tensor(0.0),
            velocity=torch.tensor([0.0, 0.0]),
            angular_velocity=torch.tensor(0.0),
            mass=torch.tensor(1.0),
            restitution=torch.tensor(0.8),
            radius=torch.tensor(0.2),
        )
        self.floor = sim2d.Floor(0.0, 0.6)
        self.shapes = [ball]


log_conf = sim2d.LoggingConfig(enable_timing=True, enable_hdf5=True, log_file="data/log.h5")
sim = MySim(sim_time=1.0, log_conf=log_conf)
sim.run()
```

## Logging

Logging is controlled through `LoggingConfig`:

- `enable_timing`: print per-block timing summaries.
- `enable_hdf5`: write simulation data to an HDF5 file.
- `enable_detailed_hdf5`: include per-Newton-step solver data.
- `log_file`: output path for the HDF5 log.

## GNN workflow

Generate training data (creates HDF5 passes under `data/gnn_datasets/raw`):

```bash
python scripts/generate_training_data.py --num_passes 100 --dataset_path data/gnn_datasets
```

Train the GNN (processes datasets on first run):

```bash
python scripts/training.py --dataset_root data/gnn_datasets
```

Run the GNN-only simulation example with a trained model:

```bash
python scripts/example_network.py
```

## Repository layout

- `src/sim2d`: core simulator, shapes, joints, logging, and GNN modules.
- `scripts`: runnable examples, dataset generation, training, and visualization utilities.
- `data`: default location for logs, datasets, and models.
