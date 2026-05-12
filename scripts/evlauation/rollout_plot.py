import matplotlib.pyplot as plt
import numpy as np

OUTPUT = "data/longer_runs/articulated.pdf"

STATS_FILES = [
    "data/longer_runs/dataset_articulated/stats_l1_new.npz",
    "data/longer_runs/dataset_articulated/stats_res_new.npz",
]

STATS_NAMES = ["MAE", "Residue"]

COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]


def plot(ax, stats, color, name):
    median, p25, p75 = stats
    ax.plot(median, color=color, linewidth=2.0, label=f"{name} Median")
    ax.fill_between(np.arange(len(p25)), p25, p75, color=color, alpha=0.2, label=f"{name} IQR")


SUBPLOTS = [
    ("trans", "Translation error (%)"),
    ("rot", "Rotation error (%)"),
]


def make_figure(key, ylabel):
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (stats_file, name) in enumerate(zip(STATS_FILES, STATS_NAMES)):
        color = COLORS[i % len(COLORS)]
        sc = np.load(stats_file)
        stats = sc[f"{key}_median"], sc[f"{key}_p25"], sc[f"{key}_p75"]
        plot(ax, stats, color, name)
    ax.set_xlabel("Simulation step", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(loc="upper left", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.tick_params(axis="both", which="minor", labelsize=14)
    plt.tight_layout()
    return fig


def main():
    from pathlib import Path

    for key, ylabel in SUBPLOTS:
        fig = make_figure(key, ylabel)
        if OUTPUT is not None:
            p = Path(OUTPUT)
            out = p.parent / f"{p.stem}_{key}{p.suffix}"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Saved plot to {out}")
        plt.show()


if __name__ == "__main__":
    main()
