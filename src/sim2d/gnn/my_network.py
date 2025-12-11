import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

from torch_geometric.nn import HeteroConv, LayerNorm
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from tqdm import tqdm
from pathlib import Path

from sim2d.gnn.dataset import DatasetSim2D


class GNNSim2D(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        pass


def train(
    model: GNNSim2D,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loss_fn: Callable,
):
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for data in pbar:
            data: HeteroData = data.to(DEVICE)
            optimizer.zero_grad()
            pred = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
            loss: torch.Tensor = loss_fn(pred, data.y)  # VERIFY LOSS CALCULATION
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        print(f"Epoch {epoch+1} Complete. Avg Loss: {total_loss/len(loader):.6f}")
        scheduler.step(epoch)
        torch.save(model.state_dict(), DATASET_ROOT / "model.pt")


def main():
    model = GNNSim2D()
    dataset = DatasetSim2D(root=DATASET_ROOT)
    loader = DataLoader(dataset, shuffle=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR_INIT, momentum=MOMENTUM)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, EPOCHS // 10, LR_GAMMA)
    loss_fn = F.mse_loss
    model.to(DEVICE)
    model.train()
    train(model, loader, optimizer, scheduler, loss_fn)


if __name__ == "__main__":
    LR_INIT = 1e-4
    LR_GAMMA = 0.5
    MOMENTUM = 0.9
    EPOCHS = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATASET_ROOT = Path("data/gnn_datasets/test_dataset")
    main()
