#!/usr/bin/env python3
"""
Plot train and validation loss CSVs written by traingpt_simple.py.

Usage:
    python3 plot_loss_csvs.py
    python3 plot_loss_csvs.py logs/<run_id>_train_loss.csv logs/<run_id>_val_loss.csv
"""

import argparse
import csv
from pathlib import Path


def read_loss_csv(path: Path, loss_column: str):
    steps = []
    losses = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = {"step", loss_column} - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path} is missing column(s): {', '.join(sorted(missing))}")
        for row in reader:
            steps.append(int(row["step"]))
            losses.append(float(row[loss_column]))
    if not steps:
        raise SystemExit(f"{path} does not contain any rows")
    return steps, losses


def plot_loss(steps, losses, title: str, output_path: Path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required to write plots. Install it with: "
            "python3 -m pip install matplotlib"
        ) from exc

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.plot(steps, losses, linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def infer_prefix(train_csv: Path) -> str:
    suffix = "_train_loss"
    return train_csv.stem[:-len(suffix)] if train_csv.stem.endswith(suffix) else "loss"


def find_latest_loss_csvs(log_dir: Path):
    train_csvs = sorted(
        log_dir.glob("*_train_loss.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for train_csv in train_csvs:
        prefix = infer_prefix(train_csv)
        val_csv = log_dir / f"{prefix}_val_loss.csv"
        if val_csv.exists():
            return train_csv, val_csv
    raise SystemExit(
        f"Could not find a matching *_train_loss.csv and *_val_loss.csv pair in {log_dir}"
    )


def main():
    parser = argparse.ArgumentParser(description="Plot train and validation loss from CSV files.")
    parser.add_argument("train_csv", nargs="?", type=Path, help="CSV with columns: step, train_loss")
    parser.add_argument("val_csv", nargs="?", type=Path, help="CSV with columns: step, val_loss")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs/e89aa444-0e2e-4253-a8d6-69833bfa305c/0.06"),
        help="Directory to scan when train_csv and val_csv are omitted.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for PNG plots. Defaults to the train CSV directory.",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix. Defaults to the run id inferred from train_csv.",
    )
    args = parser.parse_args()

    if (args.train_csv is None) != (args.val_csv is None):
        raise SystemExit("Pass both train_csv and val_csv, or omit both to use the latest logs pair.")

    if args.train_csv is None:
        train_csv, val_csv = find_latest_loss_csvs(args.log_dir)
    else:
        train_csv = args.train_csv
        val_csv = args.val_csv

    out_dir = args.out_dir or train_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or infer_prefix(train_csv)

    train_steps, train_losses = read_loss_csv(train_csv, "train_loss")
    val_steps, val_losses = read_loss_csv(val_csv, "val_loss")

    train_plot = out_dir / f"{prefix}_train_loss.png"
    val_plot = out_dir / f"{prefix}_val_loss.png"

    plot_loss(train_steps, train_losses, "Train Loss", train_plot)
    plot_loss(val_steps, val_losses, "Validation Loss", val_plot)

    print(f"Wrote {train_plot}")
    print(f"Wrote {val_plot}")


if __name__ == "__main__":
    main()
