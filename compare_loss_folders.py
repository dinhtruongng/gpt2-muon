#!/usr/bin/env python3
"""
Compare train and validation losses from two training log folders.

Each folder should contain a matching pair:
    *_train_loss.csv
    *_val_loss.csv

Usage:
    python3 compare_loss_folders.py path/to/nonorm/logs path/to/norm/logs
    python3 compare_loss_folders.py path/to/nonorm/logs path/to/norm/logs --out-dir plots
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


def infer_prefix(train_csv: Path) -> str:
    suffix = "_train_loss"
    return train_csv.stem[:-len(suffix)] if train_csv.stem.endswith(suffix) else "loss"


def find_loss_csvs(folder: Path):
    candidates = [folder]
    if (folder / "logs").is_dir():
        candidates.append(folder / "logs")

    for log_dir in candidates:
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
        f"Could not find a matching *_train_loss.csv and *_val_loss.csv pair in {folder}"
    )


def plot_comparison(series, title: str, ylabel: str, output_path: Path):
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
    colors = ["#1f77b4", "#d62728"]
    for idx, item in enumerate(series):
        ax.plot(
            item["steps"],
            item["losses"],
            label=item["label"],
            color=colors[idx % len(colors)],
            linewidth=1.8,
        )
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)
    ax.legend(title="Run")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Create combined train-loss and val-loss plots for two log folders."
    )
    parser.add_argument("nonorm_folder", type=Path, help="Folder containing NoNorm loss CSVs")
    parser.add_argument("norm_folder", type=Path, help="Folder containing Norm loss CSVs")
    parser.add_argument(
        "--labels",
        nargs=2,
        default=["NoNorm", "Norm"],
        metavar=("LABEL_A", "LABEL_B"),
        help="Legend labels for the two runs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("plots"),
        help="Directory for combined PNG plots.",
    )
    parser.add_argument(
        "--prefix",
        default="loss_compare",
        help="Output filename prefix.",
    )
    args = parser.parse_args()

    nonorm_train_csv, nonorm_val_csv = find_loss_csvs(args.nonorm_folder)
    norm_train_csv, norm_val_csv = find_loss_csvs(args.norm_folder)

    train_series = []
    val_series = []
    for label, train_csv, val_csv in [
        (args.labels[0], nonorm_train_csv, nonorm_val_csv),
        (args.labels[1], norm_train_csv, norm_val_csv),
    ]:
        train_steps, train_losses = read_loss_csv(train_csv, "train_loss")
        val_steps, val_losses = read_loss_csv(val_csv, "val_loss")
        train_series.append({"label": label, "steps": train_steps, "losses": train_losses})
        val_series.append({"label": label, "steps": val_steps, "losses": val_losses})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_plot = args.out_dir / f"{args.prefix}_train_loss.png"
    val_plot = args.out_dir / f"{args.prefix}_val_loss.png"

    plot_comparison(train_series, "Train Loss", "Loss", train_plot)
    plot_comparison(val_series, "Validation Loss", "Loss", val_plot)

    print(f"NoNorm train CSV: {nonorm_train_csv}")
    print(f"NoNorm val CSV: {nonorm_val_csv}")
    print(f"Norm train CSV: {norm_train_csv}")
    print(f"Norm val CSV: {norm_val_csv}")
    print(f"Wrote {train_plot}")
    print(f"Wrote {val_plot}")


if __name__ == "__main__":
    main()
