import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "time_eval"


def load_config(config_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all pass CSVs for a configuration, grouped by operation name."""
    raw_dir = config_dir / "raw"
    frames: dict[str, list[pd.Series]] = {}
    for csv_file in sorted(raw_dir.glob("*.csv")):
        df = pd.read_csv(csv_file, index_col=0)
        for _, row in df.iterrows():
            op = row["Operation"]
            frames.setdefault(op, []).append(row)
    return {op: pd.DataFrame(rows) for op, rows in frames.items()}


NUMERIC_COLS = ["Mean (ms)", "Median (ms)", "Std (ms)", "Min (ms)", "Max (ms)", "Calls"]
EXCLUDE_OPS = {"contacts_and_joints", "update_shapes"}


def compute_stats(op_df: pd.DataFrame) -> pd.Series:
    return op_df[NUMERIC_COLS].astype(float).mean()


def print_config_stats(config_name: str, op_stats: dict[str, pd.Series], n_passes: int) -> None:
    print(f"\n{'=' * 80}")
    print(f"  {config_name}")
    print(f"{'=' * 80}")
    header = f"  {'Operation':<25}" + "".join(f"  {c:>12}" for c in NUMERIC_COLS)
    print(header)
    print(f"  {'-' * 75}")
    for op, s in op_stats.items():
        if op in EXCLUDE_OPS:
            continue
        row = f"  {op:<25}" + "".join(f"  {s[c]:>12.3f}" for c in NUMERIC_COLS)
        print(row)
    print(f"  (averaged over {n_passes} passes)")


def main():
    configs = sorted(p for p in DATA_DIR.iterdir() if p.is_dir())
    if not configs:
        print(f"No configurations found in {DATA_DIR}")
        return

    for config_dir in configs:
        op_frames = load_config(config_dir)
        op_stats = {op: compute_stats(df) for op, df in op_frames.items()}
        n_passes = len(next(iter(op_frames.values())))
        print_config_stats(config_dir.name, op_stats, n_passes)


if __name__ == "__main__":
    main()
