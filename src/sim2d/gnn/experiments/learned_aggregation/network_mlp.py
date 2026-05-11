from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from sim2d.gnn.dataset import NODE_FEATURE_DIMS, EDGE_FEATURE_DIMS, OUTPUT_FEATURE_DIMS
from sim2d.gnn.network import MLP, Encoder, Decoder

_INCOMING: Dict[str, List[Tuple[str, str, str]]] = {}
for edge_type in EDGE_FEATURE_DIMS.keys():
    _INCOMING.setdefault(edge_type[2], []).append(edge_type)
for dst in _INCOMING:
    _INCOMING[dst].sort()


class InteractionNetworkMLP(nn.Module):
    def __init__(self, hidden_dims: int, hidden_layers: int, normalize: bool):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.mlp_edges = nn.ModuleDict(
            {
                "_".join(edge_type): MLP(
                    3 * hidden_dims, hidden_dims, hidden_dims, hidden_layers, normalize
                )
                for edge_type in EDGE_FEATURE_DIMS.keys()
            }
        )

        self.mlp_nodes = nn.ModuleDict()
        for node_type in NODE_FEATURE_DIMS.keys():
            n_incoming = len(_INCOMING.get(node_type, []))
            self.mlp_nodes[node_type] = MLP(
                hidden_dims * (1 + n_incoming),
                hidden_dims,
                hidden_dims,
                hidden_layers,
                normalize,
            )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple, torch.Tensor],
        edge_attr_dict: Dict[Tuple, torch.Tensor],
    ):
        # --- edge update ---
        new_edge_attr: Dict[Tuple, torch.Tensor] = {}
        for edge_type, edge_index in edge_index_dict.items():
            edge_attr = edge_attr_dict[edge_type]
            if edge_index.shape[1] == 0:
                new_edge_attr[edge_type] = edge_attr
                continue
            src_type, _, dst_type = edge_type
            x_src = x_dict[src_type][edge_index[0]]
            x_dst = x_dict[dst_type][edge_index[1]]
            key = "_".join(edge_type)
            new_edge_attr[edge_type] = (
                self.mlp_edges[key](torch.cat([x_src, x_dst, edge_attr], dim=-1)) + edge_attr
            )

        # --- node update ---
        new_x_dict: Dict[str, torch.Tensor] = {}
        for node_type, x in x_dict.items():
            num_nodes = x.size(0)
            parts = [x]
            for in_et in _INCOMING.get(node_type, []):
                edge_index = edge_index_dict.get(in_et)
                if edge_index is not None and edge_index.shape[1] > 0:
                    agg = scatter(
                        new_edge_attr[in_et],
                        edge_index[1],
                        dim=0,
                        dim_size=num_nodes,
                        reduce="mean",
                    )
                else:
                    agg = x.new_zeros(num_nodes, self.hidden_dims)
                parts.append(agg)
            concat = torch.cat(parts, dim=-1)
            new_x_dict[node_type] = self.mlp_nodes[node_type](concat) + x

        return new_x_dict, new_edge_attr


class ProcessorMLP(nn.Module):
    def __init__(
        self,
        message_passes: int,
        hidden_dims: int,
        hidden_layers: int,
        normalize: bool,
        repetitions: int = 1,
    ):
        super().__init__()
        self.repetitions = repetitions
        self.processor_layers = nn.ModuleList(
            [
                InteractionNetworkMLP(hidden_dims, hidden_layers, normalize)
                for _ in range(message_passes)
            ]
        )

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple, torch.Tensor],
        edge_attr_dict: Dict[Tuple, torch.Tensor],
    ):
        for _ in range(self.repetitions):
            for layer in self.processor_layers:
                x_dict, edge_attr_dict = layer(x_dict, edge_index_dict, edge_attr_dict)
        return x_dict, edge_attr_dict


class GNNSim2D(nn.Module):
    def __init__(
        self,
        message_passes: int,
        hidden_dims: int,
        hidden_layers: int,
        normalize: bool,
        stats: Dict[str, Any] = None,
        repetitions: int = 1,
    ):
        super().__init__()
        self.encoder = Encoder(hidden_dims, hidden_layers, normalize, stats=stats)
        self.processor = ProcessorMLP(
            message_passes, hidden_dims, hidden_layers, normalize, repetitions
        )
        self.decoder = Decoder(hidden_dims, hidden_layers)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict, edge_attr_dict = self.encoder(x_dict, edge_attr_dict)
        x_dict, edge_attr_dict = self.processor(x_dict, edge_index_dict, edge_attr_dict)
        nodes_dict, edges_dict = self.decoder(x_dict, edge_attr_dict)
        return nodes_dict, edges_dict
