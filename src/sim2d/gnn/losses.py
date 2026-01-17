import torch
import torch.nn.functional as F


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
        return F.l1_loss(pred_values, gt_values)

    def weighted_l1_loss(self, data, object_states, lambdas_dict) -> torch.Tensor:
        gt_values = data["object"].y.flatten()
        pred_values = object_states.flatten()
        weight = torch.ones_like(pred_values)
        if ("floor", "contact", "object") in data.edge_types:
            gt_values = torch.cat([gt_values, data[("floor", "contact", "object")].y])
            pred_values = torch.cat(
                [pred_values, lambdas_dict[("floor", "contact", "object")].flatten()]
            )
            weight = torch.cat(
                [
                    weight,
                    1 / data["object"].x[data[("floor", "contact", "object")].edge_index[1], 1],
                ]
            )
        if ("object", "contact", "object") in data.edge_types:
            gt_values = torch.cat([gt_values, data[("object", "contact", "object")].y])
            pred_values = torch.cat(
                [pred_values, lambdas_dict[("object", "contact", "object")].flatten()]
            )
            weight = torch.cat(
                [
                    weight,
                    1 / data["object"].x[data[("object", "contact", "object")].edge_index[1], 1],
                ]
            )
        return F.l1_loss(pred_values, gt_values, weight=weight)

    def residue_loss(self, data, object_states, lambdas_dict) -> torch.Tensor:
        dt = (
            torch.ones(
                (data["object"].x.shape[0], 1),
                dtype=data["object"].x.dtype,
                device=data["object"].x.device,
            )
            * self.dt
        )
        gravity = self.gravity.repeat(data["object"].x.shape[0], 1).to(data["object"].x.device)

        restitutions = data["object"].x[:, 0].unsqueeze(1)
        masses = data["object"].x[:, 1].unsqueeze(1)
        v_init = data["object"].x[:, [2, 3, 5]]
        v_pred = object_states

        res_v = v_pred - v_init - gravity * dt
        total_impulse = torch.zeros_like(res_v)
        res_c_list = []

        contact_types = [
            ("floor", "contact", "object"),
            ("object", "contact", "object"),
        ]

        for edge_type in contact_types:
            if edge_type not in data.edge_types or edge_type not in lambdas_dict:
                continue

            edge_index = data[edge_type].edge_index
            edge_attr = data[edge_type].edge_attr
            lambdas = lambdas_dict[edge_type]
            jacobians = edge_attr[:, :3]
            dists = edge_attr[:, 3].unsqueeze(1)
            target_indices = edge_index[1]

            impulse = -lambdas * jacobians
            total_impulse.index_add_(0, target_indices, impulse)

            body_restitution = restitutions[target_indices]
            body_v_init = v_init[target_indices]
            body_v_pred = v_pred[target_indices]

            b_restitution = body_restitution * body_v_init
            v_term = body_v_pred + b_restitution
            b_scaled = (jacobians * v_term).sum(dim=1, keepdim=True)
            edge_dt = dt[target_indices]
            b_error = -(self.beta / edge_dt) * dists
            a = b_scaled + b_error
            b = lambdas
            fb = a + b - torch.sqrt(a**2 + b**2 + self.eps)
            res_c_list.append(fb)

        res_v = res_v + total_impulse / masses
        res_v_flat = res_v.flatten()
        if res_c_list:
            res_c_flat = torch.cat(res_c_list).flatten()
            total_res = torch.cat([res_v_flat, res_c_flat])
        else:
            total_res = res_v_flat

        return torch.norm(total_res)
