from typing import Callable, Any, Dict, Optional, Tuple, Union, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LayerNorm, MessagePassing
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from sim2d.gnn.dataset import DatasetSim2D
from sim2d.gnn.dataset import (
    NODE_FEATURE_DIMS,
    EDGE_FEATURE_DIMS,
    OUTPUT_FEATURE_DIMS,
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

    def forward(self, x: torch.Tensor):
        return self.mlp(x)


class Encoder(nn.Module):
    def __init__(self, hidden_dims: int, hidden_layers: int, normalize: bool):
        super().__init__()
        self.mlp_nodes = nn.ModuleDict(
            {
                node_type: MLP(node_dim, hidden_dims, hidden_dims, hidden_layers, normalize)
                for node_type, node_dim in NODE_FEATURE_DIMS.items()
            }
        )
        self.mlp_edges = nn.ModuleDict(
            {
                "_".join(edge_type): MLP(
                    edge_dim, hidden_dims, hidden_dims, hidden_layers, normalize
                )
                for edge_type, edge_dim in EDGE_FEATURE_DIMS.items()
            }
        )

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_attr_dict: Dict[Tuple, torch.Tensor]):
        x_dict_encoded = {}
        for node_type, x in x_dict.items():
            x_dict_encoded[node_type] = self.mlp_nodes[node_type](x)
        edge_attr_dict_encoded = {}
        for edge_type, edge_attr in edge_attr_dict.items():
            edge_attr_dict_encoded[edge_type] = self.mlp_edges["_".join(edge_type)](edge_attr)
        return x_dict_encoded, edge_attr_dict_encoded


class Decoder(nn.Module):
    def __init__(self, hidden_dims: int, hidden_layers: int):
        super().__init__()
        self.mlp_objects = MLP(
            hidden_dims,
            OUTPUT_FEATURE_DIMS["object"],
            hidden_dims,
            hidden_layers,
            False,
        )
        self.mlp_f2o = MLP(
            hidden_dims,
            OUTPUT_FEATURE_DIMS[("floor", "contact", "object")],
            hidden_dims,
            hidden_layers,
            False,
        )
        self.mlp_o2o = MLP(
            hidden_dims,
            OUTPUT_FEATURE_DIMS[("object", "contact", "object")],
            hidden_dims,
            hidden_layers,
            False,
        )

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_attr_dict: Dict[Tuple, torch.Tensor]):
        objects_decoded = self.mlp_objects(x_dict["object"])
        contacts_decoded = {}
        if ("object", "contact", "object") in edge_attr_dict:
            contacts_decoded[("object", "contact", "object")] = self.mlp_o2o(
                edge_attr_dict[("object", "contact", "object")]
            )
        if ("floor", "contact", "object") in edge_attr_dict:
            contacts_decoded[("floor", "contact", "object")] = self.mlp_o2o(
                edge_attr_dict[("floor", "contact", "object")]
            )
        return objects_decoded, contacts_decoded


class InteractionNetwork(MessagePassing):
    def __init__(self, hidden_dims: int, hidden_layers: int, normalize: bool, aggr: str = "add"):
        super().__init__(aggr)
        self.mlp_node = MLP(2 * hidden_dims, hidden_dims, hidden_dims, hidden_layers, normalize)
        self.mlp_edge = MLP(3 * hidden_dims, hidden_dims, hidden_dims, hidden_layers, normalize)

    def edge_update(self, x_i, x_j, edge_attr):
        return self.mlp_edge(torch.cat([x_i, x_j, edge_attr], dim=-1)) + edge_attr

    def message(self, edge_attr):
        return edge_attr

    def update(self, aggr_out, x):
        if isinstance(x, tuple):
            x = x[1]
        return self.mlp_node(torch.cat([x, aggr_out], dim=-1)) + x

    def forward(
        self,
        x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ):
        if isinstance(x, tuple):
            size = (x[0].size(0), x[1].size(0))
        else:
            size = (x.size(0), x.size(0))
        edge_attr_updated = self.edge_updater(edge_index, x=x, edge_attr=edge_attr)
        x_updated = self.propagate(edge_index, x=x, edge_attr=edge_attr_updated, size=size)
        return x_updated, edge_attr_updated


class Processor(nn.Module):
    def __init__(self, message_passes: int, hidden_dims: int, hidden_layers: int, normalize: bool):
        super().__init__()
        self.processor_layers = nn.ModuleList()
        for _ in range(message_passes):
            layer_dict = nn.ModuleDict()
            for edge_type in EDGE_FEATURE_DIMS.keys():
                layer_dict["_".join(edge_type)] = InteractionNetwork(
                    hidden_dims, hidden_layers, normalize
                )
            self.processor_layers.append(layer_dict)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str], torch.Tensor],
        edge_attr_dict: Dict[Tuple[str], torch.Tensor],
    ):
        for layer in self.processor_layers:
            x_res_aggr = {key: torch.zeros_like(x) for key, x in x_dict.items()}
            for edge_type in edge_index_dict.keys():
                src_type, _, dst_type = edge_type
                edge_index = edge_index_dict[edge_type]
                edge_attr = edge_attr_dict[edge_type]
                if src_type == dst_type:
                    x = x_dict[src_type]
                else:
                    x = (x_dict[src_type], x_dict[dst_type])
                x_updated, edge_attr_updated = layer["_".join(edge_type)](x, edge_index, edge_attr)
                edge_attr_dict[edge_type] = edge_attr_updated
                x_res_aggr[dst_type] += x_updated - x_dict[dst_type]
            for node_type, x_res_aggr in x_res_aggr.items():
                x_dict[node_type] = x_dict[node_type] + x_res_aggr
        return x_dict, edge_attr_dict


class GNNSim2D(nn.Module):
    def __init__(self, message_passes: int, hidden_dims: int, hidden_layers: int, normalize: bool):
        super().__init__()
        self.encoder = Encoder(hidden_dims, hidden_layers, normalize)
        self.processor = Processor(message_passes, hidden_dims, hidden_layers, normalize)
        self.decoder = Decoder(hidden_dims, hidden_layers)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict, edge_attr_dict = self.encoder(x_dict, edge_attr_dict)
        x_dict, edge_attr_dict = self.processor(x_dict, edge_index_dict, edge_attr_dict)
        object_states, lambdas_dict = self.decoder(x_dict, edge_attr_dict)
        return object_states, lambdas_dict


def loss_fn(data, object_states, lambdas_dict):
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


def train(
    model: GNNSim2D,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
):
    for epoch in range(EPOCHS):
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for i, data in enumerate(pbar, 1):
            data: HeteroData = data.to(DEVICE)
            optimizer.zero_grad()
            states, lambdas = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
            loss: torch.Tensor = loss_fn(data, states, lambdas)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{total_loss/i:.4f}"})
        print(f"Epoch {epoch+1} Complete. Avg Loss: {total_loss/len(loader):.6f}")
        scheduler.step()
        torch.save(model.state_dict(), DATASET_ROOT / "model.pt")


def main():
    model = GNNSim2D(MESSAGE_PASSES, HIDDEN_DIMS, HIDDEN_LAYERS, NORMALIZE)
    dataset = DatasetSim2D(root=DATASET_ROOT)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_INIT)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
    model.to(DEVICE)
    model.train()
    train(model, loader, optimizer, scheduler)


if __name__ == "__main__":
    MESSAGE_PASSES = 10
    HIDDEN_LAYERS = 2
    HIDDEN_DIMS = 128
    NORMALIZE = False
    LR_INIT = 1e-3
    BATCH_SIZE = 64
    EPOCHS = 100
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATASET_ROOT = Path("data/gnn_datasets/test_dataset")
    main()
