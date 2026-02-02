from typing import Dict
import torch
import torch.nn.functional as F
from .dataset import EDGE_FEATURE_DIMS, OUTPUT_FEATURE_DIMS
from sim2d.engine import EulerSolver


class GNNLoss(torch.nn.Module):
    def __init__(
        self,
        loss_name,
        device: torch.device,
        gravity=torch.tensor([0.0, -9.81, 0.0]),
        dt=torch.tensor(0.01),
        eps=1e-6,
        beta=0.05,
    ):
        super().__init__()
        self.device = device
        if loss_name == "l1_loss":
            self.loss = self.l1_loss
        elif loss_name == "weighted_l1_loss":
            self.loss = self.weighted_l1_loss
        elif loss_name == "residue_loss":
            self.dummy_solver = EulerSolver(
                [], None, gravity, dt, None, None, self.device, None, beta, eps
            )
            self.loss = self.residue_loss
        else:
            raise NotImplementedError(f"{loss_name} loss type not found")

    def forward(self, data, object_states, lambdas_dict):
        return self.loss(data, object_states, lambdas_dict)

    def l1_loss(self, data, object_states, lambdas_dict) -> Dict[str, torch.Tensor]:
        gt_values = data["object"].y.flatten()
        pred_values = object_states.flatten()
        for edge_type in OUTPUT_FEATURE_DIMS.keys():
            if not isinstance(edge_type, tuple):
                continue
            if edge_type in data.edge_types and edge_type in lambdas_dict:
                gt_values = torch.cat([gt_values, data[edge_type].y.flatten()])
                pred_values = torch.cat([pred_values, lambdas_dict[edge_type].flatten()])
        return {"total_loss": (F.l1_loss(pred_values, gt_values), gt_values.shape[0])}

    def weighted_l1_loss(self, data, object_states, lambdas_dict) -> Dict[str, torch.Tensor]:
        gt_values = data["object"].y.flatten()
        pred_values = object_states.flatten()
        weight = torch.ones_like(pred_values)
        for edge_type in OUTPUT_FEATURE_DIMS.keys():
            if not isinstance(edge_type, tuple):
                continue
            if edge_type in data.edge_types and data[edge_type].edge_index.shape[1] > 0:
                gt_edge = data[edge_type].y.flatten()
                pred_edge = lambdas_dict[edge_type].flatten()
                gt_values = torch.cat([gt_values, gt_edge])
                pred_values = torch.cat([pred_values, pred_edge])
                target_indices = data[edge_type].edge_index[1]
                target_masses = data["object"].x[target_indices, 1]
                dim = gt_edge.numel() // target_indices.numel()
                edge_weight = (1.0 / target_masses).repeat_interleave(dim)
                weight = torch.cat([weight, edge_weight])
        return {
            "total_loss": (F.l1_loss(pred_values, gt_values, weight=weight), gt_values.shape[0])
        }

    def residue_loss(self, data, object_states, lambdas_dict) -> Dict[str, torch.Tensor]:
        device = self.device
        self.dummy_solver.restitutions = data["object"].x[:, 0]
        self.dummy_solver.inv_masses = 1 / data["object"].x[:, [1, 1, 2]]
        self.dummy_solver.num_shapes = data["object"].num_nodes
        state_init = data["object"].x[:, [3, 4, 6]]
        counts = torch.zeros(data["object"].num_nodes, dtype=torch.long, device=device)
        cons = {
            "body": [],
            "neigh": [],
            "local": [],
            "dist": [],
            "J": [],
            "Jn": [],
            "eq": [],
            "l_vals": [],
        }
        for (src, name, dst), edges in data.edge_items():
            if edges.num_edges == 0 or (src, name, dst) not in lambdas_dict:
                continue
            is_obj_obj = src == "object" and dst == "object"
            step = 2 if is_obj_obj else 1
            idx = torch.arange(0, edges.num_edges, step, device=device)
            bodies = edges.edge_index[1, idx]
            neighs = edges.edge_index[0, idx] if is_obj_obj else torch.full_like(bodies, -1)
            J_body = edges.edge_attr[idx, :3]
            J_neigh = edges.edge_attr[idx + 1, :3] if is_obj_obj else torch.zeros_like(J_body)
            dists = edges.edge_attr[idx, 3]
            preds = lambdas_dict[(src, name, dst)].view(-1)[idx]
            for i, b_idx in enumerate(bodies):
                b = b_idx.item()
                cons["body"].append(b)
                cons["neigh"].append(neighs[i].item())
                cons["local"].append(counts[b].item())
                cons["dist"].append(dists[i])
                cons["J"].append(J_body[i])
                cons["Jn"].append(J_neigh[i])
                cons["eq"].append("joint" in name)
                cons["l_vals"].append((b, counts[b].item(), preds[i]))
                counts[b] += 1
        if not cons["body"]:
            constraints = {
                "body_idx": torch.empty(0, dtype=torch.long, device=device),
                "counts": torch.zeros(data["object"].num_nodes, dtype=torch.long, device=device),
            }
        else:
            constraints = {
                "body_idx": torch.tensor(cons["body"], dtype=torch.long, device=device),
                "neighbor_idx": torch.tensor(cons["neigh"], dtype=torch.long, device=device),
                "local_idx": torch.tensor(cons["local"], dtype=torch.long, device=device),
                "dist": torch.stack(cons["dist"]),
                "jac": torch.stack(cons["J"]),
                "jac_neigh": torch.stack(cons["Jn"]),
                "is_equality": torch.tensor(cons["eq"], dtype=torch.bool, device=device),
                "counts": counts,
            }

        state = torch.zeros((len(counts), 3 + int(counts.max().item())), device=device)
        state[:, :3] = object_states
        if cons["l_vals"]:
            b, l, v = zip(*cons["l_vals"])
            state[torch.tensor(b, device=device), 3 + torch.tensor(l, device=device)] = torch.stack(
                v
            )

        res_flat = self.dummy_solver.resudial_fn(state, state_init, constraints)
        res_unflat_shape = self.dummy_solver.state_shape(constraints)
        res_unflat = res_flat.view(res_unflat_shape)
        vel_res = res_unflat[:, :3]
        body_idxs = constraints["body_idx"]
        if body_idxs.numel() > 0:
            is_equality = constraints["is_equality"]
            lambda_res_all = res_unflat[body_idxs, 3 + constraints["local_idx"]]
            joint_lambda_res = lambda_res_all[is_equality]
            contact_lambda_res = lambda_res_all[~is_equality]
        else:
            joint_lambda_res = torch.empty(0, device=device)
            contact_lambda_res = torch.empty(0, device=device)

        return {
            "total_loss": (torch.linalg.vector_norm(res_unflat, dim=1).mean(), res_unflat.shape[0]),
            "vel_res": (vel_res.abs().mean(), vel_res.shape[0]),
            "contact_res": (
                torch.nan_to_num(contact_lambda_res.abs().mean()),
                contact_lambda_res.shape[0],
            ),
            "joint_res": (
                torch.nan_to_num(joint_lambda_res.abs().mean()),
                joint_lambda_res.shape[0],
            ),
        }
