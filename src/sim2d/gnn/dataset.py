from pathlib import Path
from typing import Union, Any
import os.path as osp
import sys

import numpy as np
from tqdm import tqdm
import h5py

import torch
from torch_geometric.data import InMemoryDataset, HeteroData

NODE_FEATURE_DIMS = {"world": 4, "object": 6, "floor": 1}
EDGE_FEATURE_DIMS = {
    ("world", "w2f", "floor"): 1,
    ("world", "w2o", "object"): 4,
    ("object", "contact", "object"): 4,
    ("floor", "contact", "object"): 4,
}
OUTPUT_FEATURE_DIMS = {
    "object": 3,
    ("object", "contact", "object"): 1,
    ("floor", "contact", "object"): 1,
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

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self) -> None:
        passes_paths = []
        passes_steps = []
        raw_path = Path(self.raw_dir)
        for path in raw_path.iterdir():
            if path.suffix == ".h5":
                passes_paths.append(path)
                with h5py.File(path, "r") as f:
                    passes_steps.append(
                        len([k for k in f.keys() if k.startswith("step_")]) - 1
                    )  # cannot use last step, because we dont have prediction for it
        graphs = []
        for path_idx in tqdm(range(len(passes_paths))):
            for step_idx in range(passes_steps[path_idx]):
                with h5py.File(passes_paths[path_idx], "r") as f:
                    config = f["init_config"]
                    step = f[f"step_{step_idx:04d}"]
                    step_next = f[f"step_{step_idx+1:04d}"]
                    graph = self.construct_graph(config, step, step_next)
                    graphs.append(graph)
        self.save(graphs, self.processed_paths[0])

    def construct_graph(
        self, config: h5py.Group, step: h5py.Group, step_next: h5py.Group
    ) -> HeteroData:

        data = HeteroData()

        node_world = [
            [
                config["dt"][()],
                config["gravity"][0],
                config["gravity"][1],
                config["gravity"][2],
            ]
        ]
        data["world"].x = torch.tensor(node_world, dtype=torch.float32)

        nodes_object = []
        preds_object = []
        attrs_world_object = []
        indices_world_object = []
        for i in range(config["num_shapes"][()]):
            nodes_object.append(
                [
                    config["restitutions"][i],
                    config["masses"][i],
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
            attrs_world_object.append(
                [
                    step["shapes_data"]["translation"][i][0],
                    step["shapes_data"]["translation"][i][1],
                    norm(step["shapes_data"]["translation"][i]),
                    step["shapes_data"]["rotation"][i],
                ]
            )
            indices_world_object.append(
                [0, i],
            )
        data["object"].x = torch.tensor(nodes_object, dtype=torch.float32)
        data["object"].y = torch.tensor(preds_object, dtype=torch.float32)
        data["world", "w2o", "object"].edge_attr = torch.tensor(
            attrs_world_object, dtype=torch.float32
        )
        data["world", "w2o", "object"].edge_index = torch.tensor(
            indices_world_object, dtype=torch.long
        ).T

        if config["floor"]["active"][()]:
            data["floor"].x = torch.tensor(
                [[config["floor"]["restitution"][()]]], dtype=torch.float32
            )
            data["world", "w2f", "floor"].edge_attr = torch.tensor(
                [[config["floor"]["height"][()]]], dtype=torch.float32
            )
            data["world", "w2f", "floor"].edge_index = torch.tensor([[0, 0]], dtype=torch.long).T
        else:
            data["floor"].x = torch.zeros((0, 1), dtype=torch.float32)
            data["world", "w2f", "floor"].edge_attr = torch.zeros((0, 1), dtype=torch.float32)
            data["world", "w2f", "floor"].edge_index = torch.zeros((2, 0), dtype=torch.long)

        attrs_object_object = []
        attrs_floor_object = []
        indices_object_object = []
        indices_floor_object = []
        preds_object_object = []
        preds_floor_object = []
        object_lambda_counter = {i: 0 for i in range(config["num_shapes"][()])}
        for i in range(step["contacts_data"]["count"][()]):
            idx_1, idx_2 = step["contacts_data"]["indices"][i]
            J_1, J_2 = step["contacts_data"]["Js"][i]
            dist = step["contacts_data"]["distances"][i]
            if idx_2 != -1:
                attrs_object_object.append([J_1[0], J_1[1], J_1[2], dist])
                attrs_object_object.append([J_2[0], J_2[1], J_2[2], dist])
                indices_object_object.append([idx_2, idx_1])
                indices_object_object.append([idx_1, idx_2])
                preds_object_object.append(
                    step_next["contacts_data"]["lambdas"][idx_1][object_lambda_counter[idx_1]]
                )
                preds_object_object.append(
                    step_next["contacts_data"]["lambdas"][idx_2][object_lambda_counter[idx_2]]
                )
                object_lambda_counter[idx_1] += 1
                object_lambda_counter[idx_2] += 1
            else:
                attrs_floor_object.append([J_1[0], J_1[1], J_1[2], dist])
                indices_floor_object.append([0, idx_1])
                preds_floor_object.append(
                    step_next["contacts_data"]["lambdas"][idx_1][object_lambda_counter[idx_1]]
                )
                object_lambda_counter[idx_1] += 1

        if len(indices_object_object) > 0:
            data["object", "contact", "object"].edge_attr = torch.tensor(
                attrs_object_object, dtype=torch.float32
            )
            data["object", "contact", "object"].edge_index = torch.tensor(
                indices_object_object, dtype=torch.long
            ).T
            data["object", "contact", "object"].y = torch.tensor(
                preds_object_object, dtype=torch.float32
            )
        else:
            data["object", "contact", "object"].edge_attr = torch.zeros((0, 4), dtype=torch.float32)
            data["object", "contact", "object"].edge_index = torch.zeros((2, 0), dtype=torch.long)
            data["object", "contact", "object"].y = torch.zeros((0), dtype=torch.float32)
        if len(indices_floor_object) > 0:
            data["floor", "contact", "object"].edge_attr = torch.tensor(
                attrs_floor_object, dtype=torch.float32
            )
            data["floor", "contact", "object"].edge_index = torch.tensor(
                indices_floor_object, dtype=torch.long
            ).T
            data["floor", "contact", "object"].y = torch.tensor(
                preds_floor_object, dtype=torch.float32
            )
        else:
            data["floor", "contact", "object"].edge_attr = torch.zeros((0, 4), dtype=torch.float32)
            data["floor", "contact", "object"].edge_index = torch.zeros((2, 0), dtype=torch.long)
            data["floor", "contact", "object"].y = torch.zeros((0), dtype=torch.float32)

        return data


if __name__ == "__main__":
    DatasetSim2D(root=sys.argv[1])
