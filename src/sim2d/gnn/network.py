import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, LayerNorm
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import HeteroData


class MLP(nn.Module):
    """
    Standard MLP block for Encoders, Decoders, and Update functions.
    Structure: Linear -> ReLU -> ... -> Linear -> LayerNorm
    Follows architectural details from the paper[cite: 159, 164].
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=2, layernorm=True):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_size, output_size))
        if layernorm:
            layers.append(LayerNorm(output_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class InteractionNetworkLayer(MessagePassing):
    """
    A single message-passing layer (Processor block).
    Computes interactions (edge updates) and aggregates them to update nodes[cite: 118, 155].
    """

    def __init__(self, node_in_dim, edge_in_dim, hidden_dim):
        super(InteractionNetworkLayer, self).__init__(aggr="add")  # Sum aggregation [cite: 751]

        # Edge update function (computes messages based on src, dst, and edge attr)
        self.edge_mlp = MLP(node_in_dim * 2 + edge_in_dim, hidden_dim, hidden_dim)

        # Node update function (updates node based on aggregated messages)
        self.node_mlp = MLP(node_in_dim + hidden_dim, hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        # x is a tuple (x_src, x_dst) for bipartite/hetero graphs
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # Concatenate receiver (i), sender (j), and edge features
        tmp = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.edge_mlp(tmp)

    def update(self, aggr_out, x):
        # x[1] is the target node features
        if isinstance(x, tuple):
            x = x[1]
        # Update node state with aggregated messages
        tmp = torch.cat([x, aggr_out], dim=-1)
        return self.node_mlp(tmp)


class SimulatorGNN(nn.Module):
    def __init__(self, hidden_dim=128, num_mp_steps=4):
        super(SimulatorGNN, self).__init__()
        self.num_mp_steps = num_mp_steps

        # --- ENCODER  ---
        # Encode raw features from dataset.py into latent vectors

        # Node Encoders
        # Inputs based on DatasetSimple:
        # world: [dt, grav_x, grav_y, grav_z] -> 4
        self.node_enc_world = MLP(4, hidden_dim, hidden_dim)
        # object: [restitution, mass, vel_x, vel_y, vel_norm, ang_vel] -> 6
        self.node_enc_object = MLP(6, hidden_dim, hidden_dim)
        # floor: [restitution] -> 1
        self.node_enc_floor = MLP(1, hidden_dim, hidden_dim)

        # Edge Encoders
        # w2o: [trans_x, trans_y, trans_norm, rot] -> 4
        self.edge_enc_w2o = MLP(4, hidden_dim, hidden_dim)
        # w2f: [height] -> 1
        self.edge_enc_w2f = MLP(1, hidden_dim, hidden_dim)
        # contact: [J_x, J_y, J_z, dist] -> 4
        self.edge_enc_contact = MLP(4, hidden_dim, hidden_dim)

        # --- PROCESSOR  ---
        # Stack of M message-passing layers
        self.processor_layers = nn.ModuleList()
        for _ in range(num_mp_steps):
            conv = HeteroConv(
                {
                    ("world", "w2o", "object"): InteractionNetworkLayer(
                        hidden_dim, hidden_dim, hidden_dim
                    ),
                    ("world", "w2f", "floor"): InteractionNetworkLayer(
                        hidden_dim, hidden_dim, hidden_dim
                    ),
                    ("object", "contact", "object"): InteractionNetworkLayer(
                        hidden_dim, hidden_dim, hidden_dim
                    ),
                    ("floor", "contact", "object"): InteractionNetworkLayer(
                        hidden_dim, hidden_dim, hidden_dim
                    ),
                },
                aggr="sum",
            )
            self.processor_layers.append(conv)

        # --- DECODER  ---
        # Decode latent features back to physical quantities

        # Object Decoder: Predicts [vel_x, vel_y, ang_vel] -> 3
        self.decoder_object = MLP(hidden_dim, hidden_dim, 3, layernorm=False)

        # Contact Decoder: Predicts [lambda] -> 1
        # Takes concatenation of: Node_src (hidden) + Node_dst (hidden) + Edge_attr (hidden)
        self.decoder_contact = MLP(hidden_dim * 3, hidden_dim, 1, layernorm=False)

    def forward(self, data: HeteroData):
        # 1. Encode Nodes
        x_dict = {}
        x_dict["world"] = self.node_enc_world(data["world"].x)
        x_dict["object"] = self.node_enc_object(data["object"].x)
        if "floor" in data.node_types:
            x_dict["floor"] = self.node_enc_floor(data["floor"].x)

        # 2. Encode Edges
        edge_attr_dict = {}
        if ("world", "w2o", "object") in data.edge_types:
            edge_attr_dict[("world", "w2o", "object")] = self.edge_enc_w2o(
                data["world", "w2o", "object"].edge_attr
            )

        if ("world", "w2f", "floor") in data.edge_types:
            edge_attr_dict[("world", "w2f", "floor")] = self.edge_enc_w2f(
                data["world", "w2f", "floor"].edge_attr
            )

        if ("object", "contact", "object") in data.edge_types:
            edge_attr_dict[("object", "contact", "object")] = self.edge_enc_contact(
                data["object", "contact", "object"].edge_attr
            )

        if ("floor", "contact", "object") in data.edge_types:
            edge_attr_dict[("floor", "contact", "object")] = self.edge_enc_contact(
                data["floor", "contact", "object"].edge_attr
            )

        # 3. Processor (Message Passing)
        for conv in self.processor_layers:
            # Performs message passing on all edge types
            x_dict_out = conv(x_dict, data.edge_index_dict, edge_attr_dict)

            # Simple residual connection for node features [cite: 155]
            for key in x_dict_out:
                x_dict[key] = x_dict[key] + x_dict_out[key]

        # 4. Decoder

        # A. Object Velocity Prediction (Node-wise)
        out_object = self.decoder_object(x_dict["object"])

        # B. Contact Lambda Prediction (Edge-wise)
        # Because 'lambda' is a property of the contact edge, we must decode
        # using the final latent features of the connected nodes and the edge.
        out_contact_oo = None
        out_contact_fo = None

        if ("object", "contact", "object") in data.edge_types:
            edge_index = data["object", "contact", "object"].edge_index
            edge_latents = edge_attr_dict[("object", "contact", "object")]

            # Gather node latents for src and dst
            src, dst = edge_index
            x_src = x_dict["object"][src]
            x_dst = x_dict["object"][dst]

            # Concat [src, dst, edge] and decode
            decode_input = torch.cat([x_src, x_dst, edge_latents], dim=-1)
            out_contact_oo = self.decoder_contact(decode_input)

        if ("floor", "contact", "object") in data.edge_types:
            edge_index = data["floor", "contact", "object"].edge_index
            edge_latents = edge_attr_dict[("floor", "contact", "object")]

            src, dst = edge_index
            x_src = x_dict["floor"][src]
            x_dst = x_dict["object"][dst]

            decode_input = torch.cat([x_src, x_dst, edge_latents], dim=-1)
            out_contact_fo = self.decoder_contact(decode_input)

        return out_object, out_contact_oo, out_contact_fo
