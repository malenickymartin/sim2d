import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from tqdm import tqdm
import wandb

from sim2d import GNNSim2D, DatasetSim2D


def loss_fn(data, object_states, lambdas_dict) -> torch.Tensor:
    gt_values = data["object"].y.flatten()
    pred_values = object_states.flatten()
    if ("floor", "contact", "object") in data.edge_types:
        gt_values = torch.cat([gt_values, data[("floor", "contact", "object")].y])
        pred_values = torch.cat(
            [pred_values, lambdas_dict[("floor", "contact", "object")].flatten()]
        )
    if ("object", "contact", "object") in data.edge_types:
        gt_values = torch.cat([gt_values, data[("object", "contact", "object")].y])
        pred_values = torch.cat(
            [pred_values, lambdas_dict[("object", "contact", "object")].flatten()]
        )
    return F.mse_loss(pred_values, gt_values)


def train_epoch(
    model: GNNSim2D,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    total_epochs: int,
):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs}")
    for i, data in enumerate(pbar, 1):
        data: HeteroData = data.to(device)
        optimizer.zero_grad()
        states, lambdas = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
        loss = loss_fn(data, states, lambdas)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{total_loss/i:.4f}"})
    return total_loss / len(loader)


def validate_epoch(
    model: GNNSim2D,
    loader: DataLoader,
    device: str,
    epoch: int,
    total_epochs: int,
):
    model.eval()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{total_epochs}")
    with torch.no_grad():
        for i, data in enumerate(pbar, 1):
            data: HeteroData = data.to(device)
            states, lambdas = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
            loss = loss_fn(data, states, lambdas)
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{total_loss/i:.4f}"})
    return total_loss / len(loader)


def train(
    model: GNNSim2D,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: dict,
):
    min_val_loss = torch.inf
    for epoch in range(config["epochs"]):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            config["device"],
            epoch,
            config["epochs"],
        )
        val_loss = validate_epoch(
            model,
            val_loader,
            config["device"],
            epoch,
            config["epochs"],
        )
        print(f"Epoch {epoch+1} Complete. Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        wandb.log(
            {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "learning_rate": scheduler.get_last_lr()[0],
            }
        )
        wandb.save(str(config["dataset_root"]))
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model, config["dataset_root"] / config["model_name"])


def main(config):
    model = GNNSim2D(
        config["message_passes"],
        config["hidden_dims"],
        config["hidden_layers"],
        config["normalize"],
    )
    train_dataset = DatasetSim2D(root=config["dataset_root"] / "train_dataset")
    val_dataset = DatasetSim2D(root=config["dataset_root"] / "val_dataset")
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr_init"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, threshold=0.001)
    model.to(config["device"])
    wandb.init(project="sim2d-gnn", config=config, mode="online" if config["wandb"] else "disabled")
    wandb.config.update(
        {
            "optimizer_type": optimizer.__class__.__name__,
            "scheduler_type": scheduler.__class__.__name__,
        }
    )
    try:
        train(model, train_loader, val_loader, optimizer, scheduler, config)
    finally:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GNN Sim2D")

    parser.add_argument("--message_passes", type=int, default=5)
    parser.add_argument("--hidden_layers", type=int, default=2)
    parser.add_argument("--hidden_dims", type=int, default=128)
    parser.add_argument("--normalize", action="store_true", default=False)

    parser.add_argument("--lr_init", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dataset_root", type=Path, default=Path("data/gnn_datasets/"))
    parser.add_argument("--model_name", type=str, default="model.pt")
    parser.add_argument("--wandb", action="store_true", dest="wandb")
    parser.set_defaults(wandb=False)

    args = parser.parse_args()
    config = vars(args)

    if config["device"] is None:
        config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        config["device"] = torch.device(config["device"])
    print(f"Running on {config["device"]}.")

    main(config)
