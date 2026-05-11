from typing import Dict
import torch
import torch.nn.functional as F
from .dataset import EDGE_FEATURE_DIMS, OUTPUT_FEATURE_DIMS
from sim2d.engine import EulerSolver
from sim2d.joints import JOINT_NUM_CONSTR, JOINT_STR_TO_INT, JOINT_INT_TO_STR


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
        elif loss_name == "l1_no_lambdas":
            self.loss = self.l1_no_lambdas
        elif loss_name == "weighted_l1_loss":
            self.loss = self.weighted_l1_loss
        elif loss_name == "residue_loss":
            self.dummy_solver = EulerSolver(
                [], None, gravity, dt, None, None, self.device, None, beta, eps
            )
            self.loss = self.residue_loss
        else:
            raise NotImplementedError(f"{loss_name} loss type not found")

    def forward(self, data, object_states, joints, lambdas_dict):
        return self.loss(data, object_states, joints, lambdas_dict)

    def l1_loss(self, data, object_states, joints, lambdas_dict) -> Dict[str, torch.Tensor]:
        device = object_states.device
        gt_vel = data["object"].y.flatten()
        pred_vel = object_states.flatten()

        gt_contact_parts, pred_contact_parts = [], []
        for edge_type in OUTPUT_FEATURE_DIMS.keys():
            if not isinstance(edge_type, tuple):
                continue
            if edge_type in data.edge_types and edge_type in lambdas_dict:
                gt_contact_parts.append(data[edge_type].y.flatten())
                pred_contact_parts.append(lambdas_dict[edge_type].flatten())
        gt_contact = (
            torch.cat(gt_contact_parts) if gt_contact_parts else torch.empty(0, device=device)
        )
        pred_contact = (
            torch.cat(pred_contact_parts) if pred_contact_parts else torch.empty(0, device=device)
        )

        gt_joint = torch.empty(0, device=device)
        pred_joint = torch.empty(0, device=device)
        if joints is not None and data["joint_anchor"].num_nodes > 0:
            gt_joint = data["joint_anchor"].y.flatten()
            pred_joint = joints.flatten()

        gt_all = torch.cat([gt_vel, gt_contact, gt_joint])
        pred_all = torch.cat([pred_vel, pred_contact, pred_joint])
        return {
            "total_loss": (F.l1_loss(pred_all, gt_all), gt_all.shape[0]),
            "vel_loss": (F.l1_loss(pred_vel, gt_vel), gt_vel.shape[0]),
            "contact_loss": (
                torch.nan_to_num(F.l1_loss(pred_contact, gt_contact)),
                gt_contact.shape[0],
            ),
            "joint_loss": (torch.nan_to_num(F.l1_loss(pred_joint, gt_joint)), gt_joint.shape[0]),
        }

    def l1_no_lambdas(self, data, object_states, joints, lambdas_dict) -> Dict[str, torch.Tensor]:
        gt_values = data["object"].y.flatten()
        pred_values = object_states.flatten()
        l = F.l1_loss(pred_values, gt_values)
        return {"total_loss": (l, gt_values.shape[0]), "vel_loss": (l, gt_values.shape[0])}

    def weighted_l1_loss(
        self, data, object_states, joints, lambdas_dict
    ) -> Dict[str, torch.Tensor]:
        device = object_states.device
        gt_vel = data["object"].y.flatten()
        pred_vel = object_states.flatten()
        weight_vel = torch.ones_like(pred_vel)

        gt_contact_parts, pred_contact_parts, weight_contact_parts = [], [], []
        for edge_type in OUTPUT_FEATURE_DIMS.keys():
            if not isinstance(edge_type, tuple):
                continue
            if edge_type in data.edge_types and data[edge_type].edge_index.shape[1] > 0:
                gt_edge = data[edge_type].y.flatten()
                pred_edge = lambdas_dict[edge_type].flatten()
                gt_contact_parts.append(gt_edge)
                pred_contact_parts.append(pred_edge)
                target_indices = data[edge_type].edge_index[1]
                target_masses = data["object"].x[target_indices, 0]
                dim = gt_edge.numel() // target_indices.numel()
                weight_contact_parts.append((1.0 / target_masses).repeat_interleave(dim))
        gt_contact = (
            torch.cat(gt_contact_parts) if gt_contact_parts else torch.empty(0, device=device)
        )
        pred_contact = (
            torch.cat(pred_contact_parts) if pred_contact_parts else torch.empty(0, device=device)
        )
        weight_contact = (
            torch.cat(weight_contact_parts)
            if weight_contact_parts
            else torch.empty(0, device=device)
        )

        gt_joint = torch.empty(0, device=device)
        pred_joint = torch.empty(0, device=device)
        if joints is not None and data["joint_anchor"].num_nodes > 0:
            gt_joint = data["joint_anchor"].y.flatten()
            pred_joint = joints.flatten()
        weight_joint = torch.ones(gt_joint.numel(), device=device)

        gt_all = torch.cat([gt_vel, gt_contact, gt_joint])
        pred_all = torch.cat([pred_vel, pred_contact, pred_joint])
        weight_all = torch.cat([weight_vel, weight_contact, weight_joint])
        return {
            "total_loss": (F.l1_loss(pred_all, gt_all, weight=weight_all), gt_all.shape[0]),
            "vel_loss": (F.l1_loss(pred_vel, gt_vel, weight=weight_vel), gt_vel.shape[0]),
            "contact_loss": (
                torch.nan_to_num(F.l1_loss(pred_contact, gt_contact, weight=weight_contact)),
                gt_contact.shape[0],
            ),
            "joint_loss": (
                torch.nan_to_num(F.l1_loss(pred_joint, gt_joint, weight=weight_joint)),
                gt_joint.shape[0],
            ),
        }

    def residue_loss(self, data, object_states, joints, lambdas_dict) -> Dict[str, torch.Tensor]:
        device = self.device
        self.dummy_solver.inv_masses = 1 / data["object"].x[:, [0, 0, 1]]
        self.dummy_solver.inv_masses[torch.isinf(self.dummy_solver.inv_masses)] = 0.0
        self.dummy_solver.num_shapes = data["object"].num_nodes
        state_init = data["object"].x[:, [2, 3, 5]]
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
            "restitution": [],
        }
        for (src, name, dst), edges in data.edge_items():
            if edges.num_edges == 0 or (src, name, dst) not in lambdas_dict:
                continue
            is_obj_obj = src == "object" and dst == "object"
            is_joint = "joint" in name
            step = 2 if is_obj_obj else 1
            num_constraints = JOINT_NUM_CONSTR[JOINT_STR_TO_INT[name]] if is_joint else 1
            edge_idxs = torch.arange(0, edges.num_edges, step, device=device)

            bodies = edges.edge_index[1, edge_idxs]
            neighs = edges.edge_index[0, edge_idxs] if is_obj_obj else torch.full_like(bodies, -1)
            if is_obj_obj:
                preds_all = (
                    lambdas_dict[(src, name, dst)][edge_idxs]
                    + lambdas_dict[(src, name, dst)][edge_idxs + 1]
                ) / 2.0
            else:
                preds_all = lambdas_dict[(src, name, dst)][edge_idxs]
            for k in range(num_constraints):
                attr_offset = k * 4
                J_body = edges.edge_attr[edge_idxs, attr_offset : attr_offset + 3]
                dists = edges.edge_attr[edge_idxs, attr_offset + 3]
                if is_obj_obj:
                    J_neigh = edges.edge_attr[edge_idxs + 1, attr_offset : attr_offset + 3]
                else:
                    J_neigh = torch.zeros_like(J_body)
                preds = preds_all[:, k]
                if not is_joint:
                    restitutions = edges.edge_attr[edge_idxs, 4]

                for i, b_idx in enumerate(bodies):
                    b = b_idx.item()
                    cons["body"].append(b)
                    cons["neigh"].append(neighs[i].item())
                    cons["local"].append(counts[b].item())
                    cons["dist"].append(dists[i])
                    cons["J"].append(J_body[i])
                    cons["Jn"].append(J_neigh[i])
                    cons["eq"].append(is_joint)
                    cons["l_vals"].append((b, counts[b].item(), preds[i]))
                    if is_joint:
                        cons["restitution"].append(torch.tensor(0.0, device=device))
                    else:
                        cons["restitution"].append(restitutions[i])
                    counts[b] += 1

        # Interim joint architecture: lambdas live on joint_anchor nodes, not in lambdas_dict.
        # J data is on the incoming object/floor -> joint_anchor edges.
        if "joint_anchor" in data.node_types and data["joint_anchor"].num_nodes > 0:
            for j_name in JOINT_INT_TO_STR.values():
                n_c = JOINT_NUM_CONSTR[JOINT_STR_TO_INT[j_name]]
                obj_to_anchor_key = ("object", j_name, "joint_anchor")
                floor_to_anchor_key = ("floor", j_name, "joint_anchor")

                # Per-anchor J lookup: anchor -> [(obj_idx, attr), ...] child first then parent
                ota_map = {}
                if obj_to_anchor_key in data.edge_types and data[obj_to_anchor_key].num_edges > 0:
                    ota = data[obj_to_anchor_key]
                    for e in range(ota.num_edges):
                        a = ota.edge_index[1, e].item()
                        ota_map.setdefault(a, []).append(
                            (ota.edge_index[0, e].item(), ota.edge_attr[e])
                        )
                fta_map = {}
                if (
                    floor_to_anchor_key in data.edge_types
                    and data[floor_to_anchor_key].num_edges > 0
                ):
                    fta = data[floor_to_anchor_key]
                    for e in range(fta.num_edges):
                        fta_map[fta.edge_index[1, e].item()] = fta.edge_attr[e]

                for a in sorted(set(list(ota_map.keys()) + list(fta_map.keys()))):
                    if a >= joints.shape[0] or a not in ota_map:
                        continue
                    lambda_vals = joints[a, :n_c]
                    child_obj, child_j_attr = ota_map[a][0]
                    parent_obj = -1
                    parent_j_attr = None
                    if len(ota_map[a]) > 1:
                        parent_obj, parent_j_attr = ota_map[a][1]
                    elif a in fta_map:
                        parent_j_attr = fta_map[a]

                    b = child_obj
                    for k in range(n_c):
                        offset = k * 4
                        J_body = child_j_attr[offset : offset + 3]
                        dist = child_j_attr[offset + 3]
                        J_neigh = (
                            parent_j_attr[offset : offset + 3]
                            if parent_j_attr is not None
                            else torch.zeros(3, device=device)
                        )
                        cons["body"].append(b)
                        cons["neigh"].append(parent_obj)
                        cons["local"].append(counts[b].item())
                        cons["dist"].append(dist)
                        cons["J"].append(J_body)
                        cons["Jn"].append(J_neigh)
                        cons["eq"].append(True)
                        cons["l_vals"].append((b, counts[b].item(), lambda_vals[k]))
                        cons["restitution"].append(torch.tensor(0.0, device=device))
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
                "restitution": torch.stack(cons["restitution"]),
                "counts": counts,
            }

        state = torch.zeros((len(counts), 3 + int(counts.max().item())), device=device)
        state[:, :3] = object_states
        if cons["l_vals"]:
            b, l, v = zip(*cons["l_vals"])
            state[torch.tensor(b, device=device), 3 + torch.tensor(l, device=device)] = torch.stack(
                v
            )

        res_flat = self.dummy_solver.residual_fn(state, state_init, constraints)
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
