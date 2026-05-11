from pathlib import Path
from typing import Union, Any
import os.path as osp
import sys

import numpy as np
from tqdm import tqdm
import h5py

import torch
from torch_geometric.data import InMemoryDataset, HeteroData

from sim2d.joints import JOINT_NUM_CONSTR, JOINT_INT_TO_STR

NODE_FEATURE_DIMS = {"object": 6, "floor": 0, "joint_anchor": 0}
EDGE_FEATURE_DIMS = {
    ("object", "contact", "object"): 5,
    ("floor", "contact", "object"): 5,
    # interim joint architecture (object/floor -> joint_anchor -> object)
    ("object", "fixed_joint", "joint_anchor"): 12,
    ("object", "revolute_joint", "joint_anchor"): 8,
    ("object", "prismatic_joint", "joint_anchor"): 8,
    ("floor", "fixed_joint", "joint_anchor"): 12,
    ("floor", "revolute_joint", "joint_anchor"): 8,
    ("floor", "prismatic_joint", "joint_anchor"): 8,
    ("joint_anchor", "fixed_joint", "object"): 0,
    ("joint_anchor", "revolute_joint", "object"): 0,
    ("joint_anchor", "prismatic_joint", "object"): 0,
}
OUTPUT_FEATURE_DIMS = {
    "object": 3,
    ("object", "contact", "object"): 1,
    ("floor", "contact", "object"): 1,
    # interim joint architecture: predictions live on joint_anchor nodes, padded to max constraints
    "joint_anchor": max(JOINT_NUM_CONSTR.values()),
}


def norm(x: Any):
    return torch.norm(torch.tensor(x, dtype=torch.float32))


class DatasetSim2D(InMemoryDataset):
    """
    Args:
        root: path to directory containing HDF5 files for multiple passes
    """

    def __init__(self, root: Union[str, Path]):
        super().__init__(root)
        self.load(self.processed_paths[0], HeteroData)
        if osp.exists(self.processed_paths[1]):
            self.stats = torch.load(self.processed_paths[1])
        else:
            self.stats = None

    @property
    def processed_file_names(self):
        return ["data.pt", "statistics.pt"]

    def process(self) -> None:
        passes_paths = []
        passes_steps = []
        raw_path = Path(self.raw_dir)
        for path in sorted(raw_path.iterdir(), key=lambda p: int(p.stem.rsplit("_", 1)[1])):
            if path.suffix == ".h5":
                passes_paths.append(path)
                with h5py.File(path, "r") as f:
                    passes_steps.append(
                        len([k for k in f.keys() if k.startswith("step_")]) - 1
                    )  # cannot use last step, because we dont have prediction for it
        graphs = []
        for path_idx in tqdm(range(len(passes_paths))):
            with h5py.File(passes_paths[path_idx], "r") as f:
                config = f["init_config"]
                for step_idx in range(passes_steps[path_idx]):
                    step = f[f"step_{step_idx:04d}"]
                    step_next = f[f"step_{step_idx+1:04d}"]
                    graph = self.construct_graph(config, step, step_next)
                    graphs.append(graph)
        stats = self.calculate_statistics(graphs)

        self.save(graphs, self.processed_paths[0])
        torch.save(stats, self.processed_paths[1])

    def calculate_statistics(self, graphs):
        stats = {"nodes": {}, "edges": {}}
        for node_type in NODE_FEATURE_DIMS.keys():
            all_x = [g[node_type].x for g in graphs if g[node_type].num_nodes > 0]
            if all_x:
                all_x = torch.cat(all_x, dim=0)
                mean = all_x.mean(dim=0)
                std = all_x.std(dim=0)
                std[std < 1e-6] = 1.0
                stats["nodes"][node_type] = {"mean": mean, "std": std}

        all_edge_types = set()
        for g in graphs:
            all_edge_types.update(g.edge_types)
        for edge_type in all_edge_types:
            all_attr = [
                g[edge_type].edge_attr
                for g in graphs
                if edge_type in g.edge_attr_dict and g[edge_type].num_edges > 0
            ]
            if all_attr:
                all_attr = torch.cat(all_attr, dim=0)
                mean = all_attr.mean(dim=0)
                std = all_attr.std(dim=0)
                std[std < 1e-6] = 1.0
                stats["edges"]["_".join(edge_type)] = {"mean": mean, "std": std}
        return stats

    def construct_graph(
        self, config: h5py.Group, step: h5py.Group, step_next: h5py.Group
    ) -> HeteroData:

        data = HeteroData()

        # --- Nodes ---
        # -- Objects --
        nodes_object = []
        preds_object = []
        for i in range(config["shapes"]["num_shapes"][()]):
            nodes_object.append(
                [
                    config["shapes"]["masses"][i],
                    config["shapes"]["inertias"][i],
                    step["shapes_data"]["velocity"][i][0],
                    step["shapes_data"]["velocity"][i][1],
                    norm(step["shapes_data"]["velocity"][i]),
                    step["shapes_data"]["angular_velocity"][i],
                ]
            )
            preds_object.append(
                [
                    step_next["shapes_data"]["velocity"][i][0],
                    step_next["shapes_data"]["velocity"][i][1],
                    step_next["shapes_data"]["angular_velocity"][i],
                ]
            )
        data["object"].x = torch.tensor(nodes_object, dtype=torch.float32)
        data["object"].y = torch.tensor(preds_object, dtype=torch.float32)

        # -- Floor --
        if config["floor"]["active"][()]:
            data["floor"].x = torch.zeros((1, 0), dtype=torch.float32)
        elif (config["joints"]["num_joints"][()] > 0) and (
            config["joints"]["parent_idxs"][()] == -1
        ).any():
            data["floor"].x = torch.zeros((1, 0), dtype=torch.float32)
        else:
            data["floor"].x = torch.zeros((0, 0), dtype=torch.float32)

        # --- Edges ---
        # -- Contacts --
        attrs_object_object = []
        attrs_floor_object = []
        indices_object_object = []
        indices_floor_object = []
        preds_object_object = []
        preds_floor_object = []
        object_lambda_counter = {i: 0 for i in range(config["shapes"]["num_shapes"][()])}
        for i in range(step["contacts_data"]["count"][()]):
            idx_1, idx_2 = step["contacts_data"]["indices"][i]
            J_1, J_2 = step["contacts_data"]["Js"][i]
            dist = step["contacts_data"]["distances"][i]
            if idx_2 != -1:
                restitution = (
                    config["shapes"]["restitutions"][idx_1]
                    + config["shapes"]["restitutions"][idx_2]
                ) / 2
                attrs_object_object.append([J_1[0], J_1[1], J_1[2], dist, restitution])
                attrs_object_object.append([J_2[0], J_2[1], J_2[2], dist, restitution])
                indices_object_object.append([idx_2, idx_1])
                indices_object_object.append([idx_1, idx_2])
                preds_object_object.append(
                    [step_next["contacts_data"]["lambdas"][idx_1][object_lambda_counter[idx_1]]]
                )
                preds_object_object.append(
                    [step_next["contacts_data"]["lambdas"][idx_1][object_lambda_counter[idx_1]]]
                )
                object_lambda_counter[idx_1] += 1
            else:
                restitution = (
                    config["shapes"]["restitutions"][idx_1] + config["floor"]["restitution"][()]
                ) / 2
                attrs_floor_object.append([J_1[0], J_1[1], J_1[2], dist, restitution])
                indices_floor_object.append([0, idx_1])
                preds_floor_object.append(
                    [step_next["contacts_data"]["lambdas"][idx_1][object_lambda_counter[idx_1]]]
                )
                object_lambda_counter[idx_1] += 1
        self._assign_edge_data(
            data,
            ("object", "contact", "object"),
            indices_object_object,
            attrs_object_object,
            preds_object_object,
            EDGE_FEATURE_DIMS[("object", "contact", "object")],
            OUTPUT_FEATURE_DIMS[("object", "contact", "object")],
        )
        self._assign_edge_data(
            data,
            ("floor", "contact", "object"),
            indices_floor_object,
            attrs_floor_object,
            preds_floor_object,
            EDGE_FEATURE_DIMS[("floor", "contact", "object")],
            OUTPUT_FEATURE_DIMS[("floor", "contact", "object")],
        )

        # -- Joints --
        self.add_joints_interim(data, config, step, step_next, object_lambda_counter)

        return data

    def add_joints(self, data, config, step, step_next, object_lambda_counter):
        attrs_object_joint = {0: [], 1: [], 2: []}
        attrs_floor_joint = {0: [], 1: [], 2: []}
        indices_object_joint = {0: [], 1: [], 2: []}
        indices_floor_joint = {0: [], 1: [], 2: []}
        preds_object_joint = {0: [], 1: [], 2: []}
        preds_floor_joint = {0: [], 1: [], 2: []}
        constr_counter = 0
        for i in range(config["joints"]["num_joints"][()]):
            joint_type = config["joints"]["joint_types"][i]
            idx_1, idx_2 = config["joints"]["child_idxs"][i], config["joints"]["parent_idxs"][i]
            attrs_1, attrs_2, preds_1, preds_2 = [], [], [], []
            for _ in range(JOINT_NUM_CONSTR[joint_type]):
                J_1, J_2 = step["joint_data"]["Js"][constr_counter]
                error = step["joint_data"]["error"][constr_counter]
                constr_counter += 1
                attrs_1 += [J_1[0], J_1[1], J_1[2], error]
                preds_1.append(
                    step_next["joint_data"]["lambdas"][idx_1][object_lambda_counter[idx_1]]
                )
                if idx_2 != -1:
                    attrs_2 += [J_2[0], J_2[1], J_2[2], error]
                    preds_2.append(
                        step_next["joint_data"]["lambdas"][idx_1][object_lambda_counter[idx_1]]
                    )
                object_lambda_counter[idx_1] += 1
            if idx_2 != -1:
                attrs_object_joint[joint_type].append(attrs_1)
                attrs_object_joint[joint_type].append(attrs_2)
                indices_object_joint[joint_type].append([idx_2, idx_1])
                indices_object_joint[joint_type].append([idx_1, idx_2])
                preds_object_joint[joint_type].append(preds_1)
                preds_object_joint[joint_type].append(preds_2)
            else:
                attrs_floor_joint[joint_type].append(attrs_1)
                indices_floor_joint[joint_type].append([0, idx_1])
                preds_floor_joint[joint_type].append(preds_1)

        for i, joint_type in JOINT_INT_TO_STR.items():
            self._assign_edge_data(
                data,
                ("object", joint_type, "object"),
                indices_object_joint[i],
                attrs_object_joint[i],
                preds_object_joint[i],
                EDGE_FEATURE_DIMS[("object", joint_type, "object")],
                OUTPUT_FEATURE_DIMS[("object", joint_type, "object")],
            )
            self._assign_edge_data(
                data,
                ("floor", joint_type, "object"),
                indices_floor_joint[i],
                attrs_floor_joint[i],
                preds_floor_joint[i],
                EDGE_FEATURE_DIMS[("floor", joint_type, "object")],
                OUTPUT_FEATURE_DIMS[("floor", joint_type, "object")],
            )

    def add_joints_interim(self, data, config, step, step_next, object_lambda_counter):
        to_anchor_attrs_obj = {0: [], 1: [], 2: []}
        to_anchor_attrs_floor = {0: [], 1: [], 2: []}
        to_anchor_indices_obj = {0: [], 1: [], 2: []}
        to_anchor_indices_floor = {0: [], 1: [], 2: []}
        from_anchor_indices_obj = {0: [], 1: [], 2: []}
        max_constr = max(JOINT_NUM_CONSTR.values())
        joint_anchor_preds = []  # one row per anchor, padded to max_constr

        anchor_idx = 0
        constr_counter = 0

        for i in range(config["joints"]["num_joints"][()]):
            joint_type = config["joints"]["joint_types"][i]
            idx_1 = config["joints"]["child_idxs"][i]
            idx_2 = config["joints"]["parent_idxs"][i]

            attrs_1 = []
            attrs_2 = []
            preds = []

            for _ in range(JOINT_NUM_CONSTR[joint_type]):
                J_1, J_2 = step["joint_data"]["Js"][constr_counter]
                error = step["joint_data"]["error"][constr_counter]
                constr_counter += 1
                attrs_1 += [J_1[0], J_1[1], J_1[2], error]
                attrs_2 += [J_2[0], J_2[1], J_2[2], error]
                preds.append(
                    step_next["joint_data"]["lambdas"][idx_1][object_lambda_counter[idx_1]]
                )
                object_lambda_counter[idx_1] += 1

            # Pad preds to max_constr so all anchors share the same node y dimension
            joint_anchor_preds.append(preds + [0.0] * (max_constr - len(preds)))

            if idx_2 != -1:
                to_anchor_attrs_obj[joint_type].append(attrs_1)
                to_anchor_attrs_obj[joint_type].append(attrs_2)
                to_anchor_indices_obj[joint_type].append([idx_2, anchor_idx])
                to_anchor_indices_obj[joint_type].append([idx_1, anchor_idx])
                from_anchor_indices_obj[joint_type].append([anchor_idx, idx_1])
                from_anchor_indices_obj[joint_type].append([anchor_idx, idx_2])
            else:
                to_anchor_attrs_floor[joint_type].append(attrs_1)
                to_anchor_indices_floor[joint_type].append([0, anchor_idx])
                from_anchor_indices_obj[joint_type].append([anchor_idx, idx_1])

            anchor_idx += 1

        data["joint_anchor"].x = torch.zeros((anchor_idx, 0), dtype=torch.float32)
        if joint_anchor_preds:
            data["joint_anchor"].y = torch.tensor(joint_anchor_preds, dtype=torch.float32)
        else:
            data["joint_anchor"].y = torch.zeros((0, max_constr), dtype=torch.float32)

        for i, joint_type_str in JOINT_INT_TO_STR.items():
            attr_dim = JOINT_NUM_CONSTR[i] * 4  # J (3 values) + error per constraint
            n_to_obj = len(to_anchor_indices_obj[i])
            n_to_floor = len(to_anchor_indices_floor[i])
            n_from = len(from_anchor_indices_obj[i])

            self._assign_edge_data(
                data,
                ("object", joint_type_str, "joint_anchor"),
                to_anchor_indices_obj[i],
                to_anchor_attrs_obj[i],
                [[] for _ in range(n_to_obj)],
                attr_dim,
                0,
            )
            self._assign_edge_data(
                data,
                ("floor", joint_type_str, "joint_anchor"),
                to_anchor_indices_floor[i],
                to_anchor_attrs_floor[i],
                [[] for _ in range(n_to_floor)],
                attr_dim,
                0,
            )
            self._assign_edge_data(
                data,
                ("joint_anchor", joint_type_str, "object"),
                from_anchor_indices_obj[i],
                [[] for _ in range(n_from)],
                [
                    [] for _ in range(n_from)
                ],  # predictions live on the joint_anchor node, not the edge
                0,
                0,
            )

    def _assign_edge_data(self, data, edge_type, indices, attrs, preds, attr_dim, pred_dim):
        if len(indices) > 0:
            data[edge_type].edge_index = torch.tensor(indices, dtype=torch.long).T
            data[edge_type].edge_attr = torch.tensor(attrs, dtype=torch.float32)
            data[edge_type].y = torch.tensor(preds, dtype=torch.float32)
        else:
            data[edge_type].edge_index = torch.zeros((2, 0), dtype=torch.long)
            data[edge_type].edge_attr = torch.zeros((0, attr_dim), dtype=torch.float32)
            data[edge_type].y = torch.zeros((0, pred_dim), dtype=torch.float32)


if __name__ == "__main__":
    DatasetSim2D(root=sys.argv[1])
