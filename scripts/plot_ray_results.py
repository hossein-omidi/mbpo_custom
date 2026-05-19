#!/usr/bin/env python3
"""Parse Ray result summaries and save figures for status and trial metrics."""

import argparse
import ast
import collections
import os
import re
import sys

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise SystemExit("matplotlib is required for this script. Install it with: pip install matplotlib")

TRIAL_LINE_RE = re.compile(r"^(?P<trial_id>[^:]+(?::[^:]+)*):\s*(?P<rest>.*)$")
METRIC_RE = re.compile(r"(?P<value>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>s|iter|ts|CPUs|GPUs)")
STATUS_KEY_RE = re.compile(r"(?P<key>\w+)\s*:\s*(?P<value>.+)")


def parse_counts(text):
    """Parse the overall trial status counts from the summary line."""
    counts = collections.Counter()
    m = re.search(r"Number of trials:\s*\d+\s*\((?P<dict>.+)\)", text)
    if m:
        try:
            parsed = ast.literal_eval(m.group("dict"))
            if isinstance(parsed, dict):
                counts.update(parsed)
        except Exception:
            pass
    if not counts:
        for status in ["TERMINATED", "RUNNING", "PENDING", "FAILED", "CANCELLED"]:
            if status in text:
                counts[status] += text.count(status)
    return counts


def parse_trial_line(line):
    """Parse one line describing a trial into a dictionary."""
    line = line.strip().lstrip("-").strip()
    match = TRIAL_LINE_RE.match(line)
    if not match:
        return None
    trial_id = match.group("trial_id").strip()
    rest = match.group("rest").strip()
    pieces = [p.strip() for p in rest.split(",") if p.strip()]
    result = {"trial_id": trial_id, "status": None, "cpus": None, "gpus": None, "pid": None, "seconds": None, "iter": None, "ts": None}
    if pieces:
        result["status"] = pieces[0]
    for piece in pieces[1:]:
        if "pid" in piece:
            result["pid"] = piece
            continue
        if "CPU" in piece or "GPU" in piece:
            values = METRIC_RE.findall(piece)
            for value, unit in values:
                if unit == "CPUs":
                    result["cpus"] = int(float(value))
                elif unit == "GPUs":
                    result["gpus"] = int(float(value))
            continue
        for value, unit in METRIC_RE.findall(piece):
            if unit == "s":
                result["seconds"] = float(value)
            elif unit == "iter":
                result["iter"] = int(float(value))
            elif unit == "ts":
                result["ts"] = int(float(value))
    return result


def parse_text(text):
    """Parse the Ray output text into header info and trial records."""
    info = collections.OrderedDict()
    records = []
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    for line in lines:
        if line.startswith("=="):
            info["header"] = line.strip("= ")
            continue
        if ":" in line and not line.strip().startswith("-"):
            m = STATUS_KEY_RE.match(line)
            if m:
                key = m.group("key").strip()
                value = m.group("value").strip()
                info[key] = value
                continue
        if line.strip().startswith("-"):
            parsed = parse_trial_line(line)
            if parsed:
                records.append(parsed)
    info["status_counts"] = parse_counts(text)
    return info, records


def make_status_plot(status_counts, filename):
    labels = list(status_counts.keys())
    values = [status_counts[label] for label in labels]
    if not labels:
        return
    plt.figure(figsize=(6, 4))
    bars = plt.bar(labels, values, color="#2f72b7")
    plt.title("Trial status counts")
    plt.ylabel("Count")
    plt.xlabel("Status")
    plt.grid(axis="y", color="#cccccc", linestyle="--", alpha=0.4)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05, str(value), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def make_metrics_plot(records, filename):
    if not records:
        return
    rows = [r for r in records if r.get("seconds") is not None or r.get("iter") is not None or r.get("ts") is not None]
    if not rows:
        return
    ids = [r["trial_id"] for r in rows]
    seconds = [r.get("seconds") for r in rows]
    iters = [r.get("iter") for r in rows]
    ts = [r.get("ts") for r in rows]
    x = range(len(rows))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if any(v is not None for v in seconds):
        ax.plot(x, seconds, marker="o", label="seconds", linewidth=2)
    if any(v is not None for v in iters):
        ax.plot(x, iters, marker="s", label="iters", linewidth=2)
    if any(v is not None for v in ts):
        ax.plot(x, ts, marker="^", label="ts", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(ids, rotation=45, ha="right")
    ax.set_title("Trial metrics")
    ax.set_xlabel("Trial ID")
    ax.set_ylabel("Value")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def save_summary(info, records, filename):
    with open(filename, "w", encoding="utf-8") as out:
        for key, value in info.items():
            if key == "status_counts":
                out.write("Status counts:\n")
                for label, count in value.items():
                    out.write(f"  {label}: {count}\n")
            else:
                out.write(f"{key}: {value}\n")
        if records:
            out.write("\nTrials:\n")
            for record in records:
                out.write("  " + ", ".join(f"{k}={v}" for k, v in record.items() if v is not None) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Parse Ray summary text and create plot files.")
    parser.add_argument("input", help="Path to a Ray summary text file")
    parser.add_argument("--outdir", default=".", help="Output directory for generated images")
    parser.add_argument("--prefix", default=None, help="Prefix for output filenames")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    info, records = parse_text(text)
    os.makedirs(args.outdir, exist_ok=True)
    prefix = args.prefix or os.path.splitext(os.path.basename(args.input))[0]

    status_png = os.path.join(args.outdir, f"{prefix}_status_counts.png")
    metrics_png = os.path.join(args.outdir, f"{prefix}_trial_metrics.png")
    summary_txt = os.path.join(args.outdir, f"{prefix}_summary.txt")
    summary_svg = os.path.join(args.outdir, f"{prefix}_summary.svg")

    created = []
    if info.get("status_counts"):
        make_status_plot(info["status_counts"], status_png)
        created.append(status_png)
    if any(r.get("seconds") is not None or r.get("iter") is not None or r.get("ts") is not None for r in records):
        make_metrics_plot(records, metrics_png)
        created.append(metrics_png)
    save_summary(info, records, summary_txt)
    created.append(summary_txt)

    # Save a small SVG summary figure with the same text if requested.
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    summary_lines = [f"{k}: {v}" for k, v in info.items() if k != "status_counts"]
    summary_lines.insert(0, "Summary")
    if info.get("status_counts"):
        summary_lines.append("")
        summary_lines.extend([f"{k}: {v}" for k, v in info["status_counts"].items()])
    ax.text(0, 1, "\n".join(summary_lines), va="top", family="monospace", fontsize=10)
    plt.tight_layout()
    plt.savefig(summary_svg, dpi=150)
    plt.close()
    created.append(summary_svg)

    print("Generated:")
    for path in created:
        print("  " + path)


if __name__ == "__main__":
    main()
