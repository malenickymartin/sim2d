import torch
import torch.nn.functional as F
from .dataset import EDGE_FEATURE_DIMS, OUTPUT_FEATURE_DIMS


class GNNLoss(torch.nn.Module):
    def __init__(
        self,
        loss_name,
        gravity=torch.tensor([0.0, -9.81, 0.0]),
        dt=torch.tensor(0.01),
        eps=1e-6,
        beta=0.05,
    ):
        super().__init__()
        self.gravity = gravity
        self.dt = dt
        if loss_name == "l1_loss":
            self.loss = self.l1_loss
        elif loss_name == "weighted_l1_loss":
            self.loss = self.weighted_l1_loss
        elif loss_name == "residue_loss":
            self.eps = eps
            self.beta = beta
            self.loss = self.residue_loss
        else:
            raise NotImplementedError(f"{loss_name} loss type not found")

    def forward(self, data, object_states, lambdas_dict):
        return self.loss(data, object_states, lambdas_dict)

    def l1_loss(self, data, object_states, lambdas_dict) -> torch.Tensor:
        gt_values = data["object"].y.flatten()
        pred_values = object_states.flatten()
        for edge_type in OUTPUT_FEATURE_DIMS.keys():
            if not isinstance(edge_type, tuple):
                continue
            if edge_type in data.edge_types and edge_type in lambdas_dict:
                gt_values = torch.cat([gt_values, data[edge_type].y.flatten()])
                pred_values = torch.cat([pred_values, lambdas_dict[edge_type].flatten()])
        return F.l1_loss(pred_values, gt_values)

    def weighted_l1_loss(self, data, object_states, lambdas_dict) -> torch.Tensor:
        gt_values = data["object"].y.flatten()
        pred_values = object_states.flatten()
        weight = torch.ones_like(pred_values)
        for edge_type in OUTPUT_FEATURE_DIMS.keys():
            if not isinstance(edge_type, tuple):
                continue
            if edge_type in data.edge_types and edge_type in lambdas_dict:
                gt_edge = data[edge_type].y.flatten()
                pred_edge = lambdas_dict[edge_type].flatten()
                gt_values = torch.cat([gt_values, gt_edge])
                pred_values = torch.cat([pred_values, pred_edge])
                target_indices = data[edge_type].edge_index[1]
                target_masses = data["object"].x[target_indices, 1]
                dim = gt_edge.numel() // target_indices.numel()
                edge_weight = (1.0 / target_masses).repeat_interleave(dim)
                weight = torch.cat([weight, edge_weight])
        return F.l1_loss(pred_values, gt_values, weight=weight)

    def residue_loss(self, data, object_states, lambdas_dict) -> torch.Tensor:
        device = data["object"].x.device
        num_shapes = data["object"].x.shape[0]
        dt = torch.ones((num_shapes, 1), dtype=data["object"].x.dtype, device=device) * self.dt
        gravity = self.gravity.repeat(num_shapes, 1).to(device)

        restitutions = data["object"].x[:, 0].unsqueeze(1)
        masses = data["object"].x[:, 1]
        inertias = data["object"].x[:, 2]
        inv_masses = 1 / torch.stack([masses, masses, inertias], dim=1)
        v_init = data["object"].x[:, [3, 4, 6]]
        v_pred = object_states
        res_v = v_pred - v_init - gravity * dt

        total_vel_delta = torch.zeros_like(res_v)
        res_c_list = []
        for edge_type, num_constraints in OUTPUT_FEATURE_DIMS.items():
            if not isinstance(edge_type, tuple):
                continue
            if edge_type not in data.edge_types or edge_type not in lambdas_dict:
                continue

            edge_index = data[edge_type].edge_index
            edge_attr = data[edge_type].edge_attr
            target_indices = edge_index[1]
            lambdas_all = lambdas_dict[edge_type]
            is_contact = "contact" in edge_type[1]
            is_equality = not is_contact

            target_indices = edge_index[1]
            target_inv_masses = inv_masses[target_indices]
            target_dt = dt[target_indices]
            target_v_pred = v_pred[target_indices]

            for k in range(num_constraints):
                jacobians = edge_attr[:, k * 4 : k * 4 + 3]
                dists = edge_attr[:, k * 4 + 3].unsqueeze(1)
                if num_constraints == 1:
                    lambdas = lambdas_all
                else:
                    lambdas = lambdas_all[:, k].unsqueeze(1)
                force_impulse = -lambdas * jacobians
                vel_delta = force_impulse * target_inv_masses
                total_vel_delta.index_add_(0, target_indices, vel_delta)
                if is_equality:
                    v_term = target_v_pred
                else:
                    target_v_init = v_init[target_indices]
                    target_restitution = restitutions[target_indices]
                    b_rest_val = target_restitution * target_v_init
                    v_term = target_v_pred + b_rest_val
                b_scaled = (jacobians * v_term).sum(dim=1, keepdim=True)
                b_error = -(self.beta / target_dt) * dists
                a = b_scaled + b_error
                if is_equality:
                    res_c_list.append(a)
                else:
                    b = lambdas
                    fb = a + b - torch.sqrt(a**2 + b**2 + self.eps)
                    res_c_list.append(fb)

        res_v = res_v + total_vel_delta
        res_v_flat = res_v.flatten()
        if res_c_list:
            res_c_flat = torch.cat(res_c_list).flatten()
            total_res = torch.cat([res_v_flat, res_c_flat])
        else:
            total_res = res_v_flat

        return torch.norm(total_res)
