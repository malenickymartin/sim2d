from typing import Callable, Any, Dict, Optional, Tuple, Union, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    HeteroConv,
    LayerNorm,
    HeteroLayerNorm,
    MessagePassing,
)
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from sim2d.gnn.dataset import DatasetSim2D
from sim2d.gnn.dataset import (
    NODE_FEATURE_DIMS,
    EDGE_FEATURE_DIMS,
    OUTPUT_FEATURE_DIMS,
    EDGE_TYPES_FULL,
)


class MLP(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        hidden_dims: int = 128,
        hidden_layers: int = 2,
        output_norm: bool = True,
    ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dims, hidden_dims))
        layers.append(nn.ReLU())
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dims, hidden_dims))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dims, output_dims))
        if output_norm:
            layers.append(LayerNorm(output_dims))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class Encoder(nn.Module):
    def __init__(self, hidden_dims: int, hidden_layers: int, normalize: bool):
        super().__init__()
        self.mlp_nodes = nn.ModuleDict(
            {
                "world": MLP(
                    NODE_FEATURE_DIMS["world"], hidden_dims, hidden_dims, hidden_layers, normalize
                ),
                "floor": MLP(
                    NODE_FEATURE_DIMS["floor"], hidden_dims, hidden_dims, hidden_layers, normalize
                ),
                "object": MLP(
                    NODE_FEATURE_DIMS["object"], hidden_dims, hidden_dims, hidden_layers, normalize
                ),
            }
        )
        self.mlp_edges = nn.ModuleDict(
            {
                "w2f": MLP(
                    EDGE_FEATURE_DIMS["w2f"], hidden_dims, hidden_dims, hidden_layers, normalize
                ),
                "w2o": MLP(
                    EDGE_FEATURE_DIMS["w2o"], hidden_dims, hidden_dims, hidden_layers, normalize
                ),
                "contact": MLP(
                    EDGE_FEATURE_DIMS["contact"], hidden_dims, hidden_dims, hidden_layers, normalize
                ),
            }
        )

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_attr_dict: Dict[Tuple, torch.Tensor]):
        for key, x in x_dict.items():
            x_dict[key] = self.mlp_nodes[key](x)
        for key, e in edge_attr_dict.items():
            edge_attr_dict[key] = self.mlp_edges[key[1]](e)


class Decoder(nn.Module):
    def __init__(self, hidden_dims: int, hidden_layers: int):
        super().__init__()
        self.mlp_objects = MLP(
            NODE_FEATURE_DIMS["object"],
            OUTPUT_FEATURE_DIMS["object"],
            hidden_dims,
            hidden_layers,
            False,
        )
        self.mlp_contacts = MLP(
            EDGE_FEATURE_DIMS["contact"],
            OUTPUT_FEATURE_DIMS["contact"],
            hidden_dims,
            hidden_layers,
            False,
        )

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_attr_dict: Dict[Tuple, torch.Tensor]):
        x_dict["object"] = self.mlp_objects(x_dict["object"])
        if ("object", "contact", "object") in edge_attr_dict:
            edge_attr_dict[("object", "contact", "object")] = self.mlp_contacts(
                edge_attr_dict[("object", "contact", "object")]
            )
        if ("floor", "contact", "object") in edge_attr_dict:
            edge_attr_dict[("floor", "contact", "object")] = self.mlp_contacts(
                edge_attr_dict[("floor", "contact", "object")]
            )


class MessagePasser(MessagePassing):
    def __init__(self, hidden_dims: int, hidden_layers: int, normalize: bool, aggr: str = "mean"):
        super().__init__(aggr)
        self.mlp_message = MLP(3 * hidden_dims, hidden_dims, hidden_dims, hidden_layers, normalize)
        self.mlp_node = MLP(2 * hidden_dims, hidden_dims, hidden_dims, hidden_layers, normalize)

    def edge_update(self):
        return super().edge_update()

    def message(self, x_j, x_i, edge_attr):
        return self.mlp_message(torch.cat([x_i, x_j, edge_attr], dim=-1))

    def update(self, inputs):
        self.mlp_node()

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)


class Processor(nn.Module):
    def __init__(self, message_passes: int, hidden_layers, hidden_dims: int, normalize: bool):
        super().__init__()
        self.processor_layers = nn.ModuleList()
        for _ in range(message_passes):
            conv = HeteroConv(
                {
                    edge_type: MessagePasser(hidden_dims, hidden_layers, normalize)
                    for edge_type in EDGE_TYPES_FULL
                },
                aggr=None,
            )
            self.processor_layers.append(conv)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        for layer in self.processor_layers:
            x_dict, edge_attr_dict = layer(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
        return x_dict, edge_attr_dict


class GNNSim2D(nn.Module):
    def __init__(self, message_passes, hidden_dims, hidden_layers, normalize):
        super().__init__()
        self.encoder = Encoder(hidden_dims, hidden_layers, normalize)
        self.decoder = Decoder(hidden_dims, hidden_layers)
        self.processor = Processor(message_passes, hidden_dims, hidden_layers, normalize)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        self.encoder(x_dict, edge_attr_dict)
        self.processor(x_dict, edge_index_dict, edge_attr_dict)
        self.decoder(x_dict, edge_attr_dict)
        return


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
    model = GNNSim2D(MESSAGE_PASSES, HIDDEN_DIMS, HIDDEN_LAYERS, NORMALIZE)
    dataset = DatasetSim2D(root=DATASET_ROOT)
    loader = DataLoader(dataset, shuffle=True)
    data = next(iter(loader))
    model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
    optimizer = torch.optim.SGD(model.parameters(), lr=LR_INIT, momentum=MOMENTUM)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, EPOCHS // 10, LR_GAMMA)
    loss_fn = F.mse_loss
    model.to(DEVICE)
    model.train()
    train(model, loader, optimizer, scheduler, loss_fn)


if __name__ == "__main__":
    MESSAGE_PASSES = 10
    HIDDEN_LAYERS = 2
    HIDDEN_DIMS = 128
    NORMALIZE = False
    LR_INIT = 1e-4
    LR_GAMMA = 0.5
    MOMENTUM = 0.9
    EPOCHS = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATASET_ROOT = Path("data/gnn_datasets/test_dataset")
    main()
