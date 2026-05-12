import argparse
from collections import defaultdict
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from sim2d.gnn import DatasetSim2D, GNNLoss
from sim2d.gnn.dataset import OUTPUT_FEATURE_DIMS


def baseline_predict(data, device):
    # Input velocities: data["object"].x columns are [mass, inertia, vx, vy, |v|, ang_vel]
    object_states = data["object"].x[:, [2, 3, 5]].to(device)
    g = torch.zeros_like(object_states)
    g[:, 2] = -9.81
    object_states = object_states + 0.01 * g

    lambdas_dict = {}
    for edge_type in OUTPUT_FEATURE_DIMS:
        if isinstance(edge_type, tuple) and edge_type in data.edge_types:
            lambdas_dict[edge_type] = torch.zeros_like(data[edge_type].y, device=device)

    return object_states, lambdas_dict


def evaluate(loss_fn, loader, device):
    total_losses = defaultdict(float)
    loss_counts = defaultdict(int)

    with torch.no_grad():
        for data in tqdm(loader, desc="Evaluating baseline"):
            data = data.to(device)
            object_states, lambdas_dict = baseline_predict(data, device)
            loss_dict = loss_fn.loss(data, object_states, None, lambdas_dict)
            for k, (val, count) in loss_dict.items():
                total_losses[k] += val.item() * count
                loss_counts[k] += count

    return {k: v / loss_counts[k] if loss_counts[k] else 0.0 for k, v in total_losses.items()}


def main():
    parser = argparse.ArgumentParser(description="Baseline evaluation (input vel + zero impulses)")
    parser.add_argument(
        "--dataset_root", type=Path, default=Path("data/one_step_merge_dataset/val_dataset")
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="l1_loss",
        choices=["l1_loss", "l1_no_lambdas", "weighted_l1_loss", "residue_loss"],
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    dataset = DatasetSim2D(root=args.dataset_root)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    loss_fn = GNNLoss(args.loss_type, device)

    results = evaluate(loss_fn, loader, device)

    print(f"\nBaseline results ({args.loss_type}):")
    for k, v in results.items():
        print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()
