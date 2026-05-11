from typing import Dict, List, Tuple, Union, Any

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from sim2d.gnn.dataset import (
    NODE_FEATURE_DIMS,
    EDGE_FEATURE_DIMS,
    OUTPUT_FEATURE_DIMS,
)
from sim2d.gnn.network import MLP, Encoder, Decoder


class EdgeNetwork(MessagePassing):
    def __init__(self, hidden_dims: int, hidden_layers: int, normalize: bool, aggr: str = "add"):
        super().__init__(aggr)
        self.mlp = MLP(3 * hidden_dims, hidden_dims, hidden_dims, hidden_layers, normalize)

    def edge_update(self, x_i, x_j, edge_attr):
        return self.mlp(torch.cat([x_i, x_j, edge_attr], dim=-1)) + edge_attr

    def message(self, edge_attr):
        return edge_attr

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
        x_aggr = self.propagate(edge_index, edge_attr=edge_attr_updated, size=size)
        return x_aggr, edge_attr_updated


class NodeNetwork(nn.Module):
    def __init__(
        self,
        edge_types: List[Tuple],
        hidden_dims: int,
        hidden_layers: int,
        normalize: bool,
        num_heads: int = 4,
    ):
        super().__init__()
        self.edge_types = edge_types
        self.hidden_dims = hidden_dims
        self.attention = nn.MultiheadAttention(hidden_dims, num_heads, batch_first=True)
        self.mlp = MLP(hidden_dims, hidden_dims, hidden_dims, hidden_layers, normalize, False)

    def forward(
        self,
        x: torch.Tensor,
        aggr_dict: Dict[Tuple, torch.Tensor],
        n_nodes: int,
    ):
        parts = []
        for edge_type in self.edge_types:
            if edge_type in aggr_dict:
                parts.append(aggr_dict[edge_type])
            else:
                parts.append(torch.zeros(n_nodes, self.hidden_dims, device=x.device))
        kv = torch.stack(parts, dim=1)  # [n_nodes, n_edge_types, hidden_dims]
        attn_out, _ = self.attention(x.unsqueeze(1), kv, kv, need_weights=False)
        return self.mlp(attn_out.squeeze(1)) + x


class Processor(nn.Module):
    def __init__(
        self,
        message_passes: int,
        hidden_dims: int,
        hidden_layers: int,
        normalize: bool,
        num_heads: int = 4,
    ):
        super().__init__()

        edge_types_per_node: Dict[str, List[Tuple]] = {}
        for edge_type in EDGE_FEATURE_DIMS.keys():
            edge_types_per_node.setdefault(edge_type[2], []).append(edge_type)

        self.processor_layers = nn.ModuleList()
        for _ in range(message_passes):
            layer_dict = nn.ModuleDict()
            for edge_type in EDGE_FEATURE_DIMS.keys():
                layer_dict["_".join(edge_type)] = EdgeNetwork(hidden_dims, hidden_layers, normalize)
            for node_type, edge_types in edge_types_per_node.items():
                layer_dict[node_type] = NodeNetwork(
                    edge_types, hidden_dims, hidden_layers, normalize, num_heads
                )
            self.processor_layers.append(layer_dict)

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str], torch.Tensor],
        edge_attr_dict: Dict[Tuple[str], torch.Tensor],
    ):
        for layer in self.processor_layers:
            aggr_per_node: Dict[str, Dict[Tuple, torch.Tensor]] = {}

            for edge_type in edge_index_dict.keys():
                src_type, _, dst_type = edge_type
                edge_index = edge_index_dict[edge_type]
                if edge_index.shape[1] == 0:
                    continue
                edge_attr = edge_attr_dict[edge_type]
                if src_type == dst_type:
                    x = x_dict[src_type]
                else:
                    x = (x_dict[src_type], x_dict[dst_type])
                x_aggr, edge_attr_updated = layer["_".join(edge_type)](x, edge_index, edge_attr)
                edge_attr_dict[edge_type] = edge_attr_updated
                aggr_per_node.setdefault(dst_type, {})[edge_type] = x_aggr

            for node_type, x in x_dict.items():
                if node_type in layer:
                    x_dict[node_type] = layer[node_type](
                        x, aggr_per_node.get(node_type, {}), x.size(0)
                    )

        return x_dict, edge_attr_dict


class GNNSim2D(nn.Module):
    def __init__(
        self,
        message_passes: int,
        hidden_dims: int,
        hidden_layers: int,
        normalize: bool,
        num_heads: int = 4,
        stats: Dict[str, Any] = None,
    ):
        super().__init__()
        self.encoder = Encoder(hidden_dims, hidden_layers, normalize, stats=stats)
        self.processor = Processor(message_passes, hidden_dims, hidden_layers, normalize, num_heads)
        self.decoder = Decoder(hidden_dims, hidden_layers)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        x_dict, edge_attr_dict = self.encoder(x_dict, edge_attr_dict)
        x_dict, edge_attr_dict = self.processor(x_dict, edge_index_dict, edge_attr_dict)
        object_states, lambdas_dict = self.decoder(x_dict, edge_attr_dict)
        return object_states, lambdas_dict
