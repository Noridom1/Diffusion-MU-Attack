"""Plot loss progress for TextGrad attack runs.

The JSON logger used by this repository writes one record for the original
prompt and then attack/evaluation records.  Baseline TextGrad records contain
one aggregate ``loss`` per sampled diffusion timestep.  TextGrad-EOT records
contain a ``loss_window`` with one entry per optimizer update; this script
expands those windows so the x-axis is the actual optimizer step.

Example
-------
    python utils/plot_attack_loss.py \
        --root files/downloads/nudity_n20_sample2024_run0_eval_bundle/files/results/nudity_n20_sample2024_run0/unlearndiff \
        --output files/downloads/nudity_n20_sample2024_run0_eval_bundle/loss_progress.png

Use ``--max-steps 0`` to plot every logged timestep.  The default is 50 so
that the x-axis covers all 50 sampled-timestep records (0--49).
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


_ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _attack_index(path: Path) -> int:
    """Return the numeric attack index encoded in ``attack_idx_N``."""

    try:
        return int(path.name.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid attack directory name: {path.name!r}") from exc


def _parse_process_log(path: Path) -> list[dict[str, Any]]:
    """Best-effort fallback for an interrupted run without a valid log.json."""

    records: list[dict[str, Any]] = []
    if not path.is_file():
        return records

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        marker = "logging:"
        if marker not in raw_line:
            continue
        payload = _ANSI_ESCAPE.sub("", raw_line.split(marker, 1)[1]).strip()
        try:
            value = ast.literal_eval(payload)
        except (SyntaxError, ValueError):
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def load_records(attack_dir: Path) -> list[dict[str, Any]]:
    """Load logger records from one attack directory."""

    log_path = attack_dir / "log.json"
    if log_path.is_file():
        try:
            value = json.loads(log_path.read_text(encoding="utf-8"))
            if isinstance(value, list):
                return [entry for entry in value if isinstance(entry, dict)]
        except (json.JSONDecodeError, OSError):
            pass
    return _parse_process_log(attack_dir / "process.log")


def _finite_loss_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep attack records with a finite numeric loss."""

    result: list[dict[str, Any]] = []
    for record in records:
        value = record.get("loss")
        try:
            loss = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(loss):
            # Keep a normalized numeric value for callers and plotting.
            normalized = dict(record)
            normalized["loss"] = loss
            result.append(normalized)
    return result


def _loss_series(records: list[dict[str, Any]]) -> tuple[list[float], list[int], str]:
    """Extract baseline or EOT losses from logger records.

    Returns ``(losses, x_values, mode)``. EOT windows are deduplicated by
    optimizer step so a partially resumed log remains plot-safe.
    """

    eot_by_step: dict[int, float] = {}
    for record in records:
        window = record.get("loss_window")
        if not isinstance(window, list):
            continue
        for entry in window:
            if not isinstance(entry, dict):
                continue
            try:
                step = int(entry["optimization_step"])
                loss = float(entry["mean_loss"])
            except (KeyError, TypeError, ValueError):
                continue
            if step > 0 and np.isfinite(loss):
                eot_by_step[step] = loss
    if eot_by_step:
        x_values = sorted(eot_by_step)
        return (
            [eot_by_step[step] for step in x_values],
            x_values,
            "EOT optimizer step",
        )

    baseline = _finite_loss_records(records)
    return (
        [record["loss"] for record in baseline],
        list(range(len(baseline))),
        "Outer sampled-timestep record (0-based)",
    )


def _status(records: list[dict[str, Any]], attack_records: list[dict[str, Any]]) -> str:
    """Describe whether an attack succeeded, failed, or never ran."""

    if not attack_records:
        if records and bool(records[0].get("success")):
            return "baseline success; attack not run"
        return "no loss records"

    success_positions = [
        i for i, record in enumerate(attack_records) if bool(record.get("success"))
    ]
    if success_positions:
        if any(isinstance(record.get("loss_window"), list) for record in records):
            step = attack_records[success_positions[0]].get("optimization_step")
            return f"success @ optimizer step {step}"
        # Baseline TextGrad records are outer sampled-timestep records, not the
        # inner optimizer iteration (which is configured separately, usually 40).
        return f"success @ outer step {success_positions[0]}"
    return "no detector success"


def _discover_attack_dirs(root: Path, indices: list[int] | None) -> list[Path]:
    """Find attack directories and sort them numerically."""

    if not root.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {root}")

    # Allow passing either the bundle's ``.../results/...`` directory or the
    # bundle root directly.  The latter contains the results directory several
    # levels below ``files/``.
    if not any(root.glob("attack_idx_*")):
        direct_candidate = root / "unlearndiff"
        if direct_candidate.is_dir():
            root = direct_candidate
        else:
            candidates = [
                path
                for path in root.rglob("unlearndiff")
                if path.is_dir() and any(path.glob("attack_idx_*"))
            ]
            if len(candidates) == 1:
                root = candidates[0]
            elif len(candidates) > 1:
                raise ValueError(
                    "More than one unlearndiff results directory found; "
                    "pass the exact directory with --root"
                )

    if indices is None:
        paths = [path for path in root.glob("attack_idx_*") if path.is_dir()]
    else:
        paths = [root / f"attack_idx_{index}" for index in indices]
        missing = [path for path in paths if not path.is_dir()]
        if missing:
            missing_text = ", ".join(path.name for path in missing)
            raise FileNotFoundError(f"Missing attack directories: {missing_text}")

    if not paths:
        raise FileNotFoundError(f"No attack_idx_* directories found under {root}")
    return sorted(paths, key=_attack_index)


def _palette(size: int) -> list[tuple[float, float, float, float]]:
    """Return a repeatable set of saturated, high-contrast colors."""

    # tab20 supplies 20 well-separated colors and remains readable on a white
    # background.  Interleave hues so neighboring legend entries do not share
    # the light/dark variants of one hue.
    colors = list(plt.get_cmap("tab20").colors)
    order = [0, 6, 12, 18, 2, 8, 14, 4, 10, 16, 1, 7, 13, 19, 3, 9, 15, 5, 11, 17]
    selected = [colors[order[i % len(order)]] for i in range(size)]
    return [(*color, 1.0) for color in selected]


def plot_attack_losses(
    root: Path,
    output: Path,
    *,
    indices: list[int] | None = None,
    max_steps: int = 50,
    dpi: int = 180,
    log_y: bool = False,
) -> list[dict[str, Any]]:
    """Create the loss-progress plot and return the plotted series metadata."""

    if max_steps < 0:
        raise ValueError("max_steps must be non-negative (use 0 for all steps)")

    attack_dirs = _discover_attack_dirs(root, indices)
    series: list[dict[str, Any]] = []
    for attack_dir in attack_dirs:
        records = load_records(attack_dir)
        attack_records = _finite_loss_records(records)
        losses, x_values, x_label = _loss_series(records)
        original_length = len(losses)
        if max_steps:
            losses = losses[:max_steps]
            x_values = x_values[:max_steps]
        series.append(
            {
                "index": _attack_index(attack_dir),
                "losses": losses,
                "x_values": x_values,
                "x_label": x_label,
                "original_length": original_length,
                "status": _status(records, attack_records),
            }
        )

    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=dpi)
    colors = _palette(len(series))
    legend_handles: list[Line2D] = []

    for color, item in zip(colors, series):
        losses = item["losses"]
        index = item["index"]
        status = item["status"]
        if item["original_length"] > len(losses):
            status += "; clipped"
        label = f"idx {index} ({status})"
        x_values = np.asarray(item["x_values"], dtype=int)

        if losses:
            y_values = np.asarray(losses, dtype=float)
            if log_y:
                y_values = np.where(y_values > 0, y_values, np.nan)
            ax.plot(
                x_values,
                y_values,
                color=color,
                linewidth=1.8,
                alpha=0.95,
                marker="o",
                markersize=2.8,
                markevery=max(1, len(losses) // 12),
                label=label,
            )
        # An empty handle keeps baseline-success indices visible in the legend
        # even though those attacks returned before any optimization step.
        legend_handles.append(Line2D([], [], color=color, linewidth=2.2, label=label))

    x_end = max(
        (max(item["x_values"]) if item["x_values"] else 0 for item in series),
        default=1,
    )
    # Keep the requested 50-step frame even when an attack terminates early.
    x_end = max(49, x_end)
    ax.set_xlim(0, x_end)
    # Include the requested endpoint explicitly; matplotlib's automatic tick
    # locator otherwise commonly stops at 35 when the axis ends at 39.
    if x_end <= 50:
        ax.set_xticks(sorted(set([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, x_end])))
    labels = {item["x_label"] for item in series}
    x_label = labels.pop() if len(labels) == 1 else "Optimization / sampled-timestep record"
    ax.set_xlabel(x_label)
    ax.set_ylabel("Reported loss")
    ax.set_title("TextGrad attack loss progress by attack index")
    ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.35)
    if log_y:
        ax.set_yscale("log")
    ax.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        ncol=2,
        fontsize=8,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=0.96,
        title="Attack index / status",
    )
    fig.tight_layout(rect=(0, 0, 0.78, 1))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return series


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        required=True,
        help="Directory containing attack_idx_* result directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <root>/loss_progress.png).",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Optional attack indices to plot; default is every attack_idx_* directory.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Number of plotted records per line; 0 plots all records (default: 50).",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Output resolution (default: 180).")
    parser.add_argument("--log-y", action="store_true", help="Use a logarithmic y-axis.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root
    output = args.output or (root / "loss_progress.png")
    try:
        series = plot_attack_losses(
            root,
            output,
            indices=args.indices,
            max_steps=args.max_steps,
            dpi=args.dpi,
            log_y=args.log_y,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Saved plot: {output}")
    for item in series:
        print(
            f"idx {item['index']:>2}: {len(item['losses']):>2} points "
            f"({item['status']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
