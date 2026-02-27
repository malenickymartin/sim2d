import argparse
import torch
from pathlib import Path
from torch_geometric.loader import DataLoader
import wandb

from sim2d.gnn import GNNSim2D, DatasetSim2D, GNNLoss
from training import train


def sweep_train():
    wandb.init()
    sweep_config = wandb.config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_root = Path("data/one_step_merge_dataset/")

    config = {
        "device": device,
        "dataset_root": dataset_root,
        "model_name": None,
        "loss_type": "residue_loss",
        "epochs": 100,
        "wandb": "sweep",
        "message_passes": sweep_config.message_passes,
        "hidden_layers": sweep_config.hidden_layers,
        "hidden_dims": sweep_config.hidden_dims,
        "batch_size": sweep_config.batch_size,
        "lr_init": sweep_config.lr_init,
        "normalize": sweep_config.normalize,
        "normalize_input": sweep_config.normalize_input,
        "optimizer": sweep_config.optimizer,
        "weight_decay": sweep_config.weight_decay,
    }

    loss_fn = GNNLoss(config["loss_type"], config["device"])
    train_dataset = DatasetSim2D(root=config["dataset_root"] / "train_dataset")
    val_dataset = DatasetSim2D(root=config["dataset_root"] / "val_dataset")

    model = GNNSim2D(
        config["message_passes"],
        config["hidden_dims"],
        config["hidden_layers"],
        config["normalize"],
        stats=train_dataset.stats if config["normalize_input"] else None,
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    if config["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["lr_init"], weight_decay=config["weight_decay"]
        )
    elif config["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["lr_init"], weight_decay=config["weight_decay"]
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            momentum=0.9,
            weight_decay=config["weight_decay"],
            lr=config["lr_init"],
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=1e-6
    )

    model.to(config["device"])

    wandb.config.update(
        {
            "optimizer_type": optimizer.__class__.__name__,
            "scheduler_type": scheduler.__class__.__name__,
        }
    )

    try:
        train(model, loss_fn, train_loader, val_loader, optimizer, scheduler, config)
    except Exception as e:
        print(f"Run failed with error: {e}")


if __name__ == "__main__":
    sweep_configuration = {
        "method": "bayes",
        "name": "sim2d-gnn-sweep-comprehensive",
        "metric": {"goal": "minimize", "name": "val/total_loss"},
        "early_terminate": {"type": "hyperband", "min_iter": 5, "eta": 3},
        "parameters": {
            "lr_init": {"max": 0.01, "min": 0.0001},
            "batch_size": {"values": [16, 32, 64, 128]},
            "hidden_dims": {"values": [64, 128, 256]},
            "hidden_layers": {"values": [2, 4, 8, 10]},
            "message_passes": {"values": [2, 4, 8, 10]},
            "normalize": {"values": [True, False]},
            "normalize_input": {"values": [True, False]},
            "optimizer": {"values": ["SGD", "Adam", "AdamW"]},
            "weight_decay": {"values": [0.0, 1e-3, 1e-4, 1e-5]},
        },
    }

    parser = argparse.ArgumentParser(description="Run W&B Sweep Agent")
    parser.add_argument("--sweep_id", type=str, default=None)
    args = parser.parse_args()

    project_name = "sim2d-gnn"
    if args.sweep_id is None:
        sweep_id = wandb.sweep(sweep_configuration, project=project_name)
        print(f"\n=== Created NEW Sweep: {sweep_id} ===")
    else:
        sweep_id = args.sweep_id
        print(f"\n=== Joining EXISTING Sweep: {sweep_id} ===")

    wandb.agent(sweep_id, function=sweep_train, project=project_name)
