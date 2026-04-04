import re
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_report(filename):
    with open(filename, "r") as f:
        lines = [l.rstrip("\n") for l in f.readlines()]

    envs = {}
    current_env = None
    headers = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Detect environment block
        if line.startswith("==="):
            if i + 1 < len(lines):
                current_env = lines[i + 1].strip()
                envs[current_env] = {"points": {}, "series": {}}
            i += 2
            continue

        if not current_env:
            i += 1
            continue

        if line.strip().startswith("Method"):
            parts = re.split(r"\s{2,}", line.strip())
            headers = parts[1:]
            i += 1
            continue

        if line.startswith("---"):
            i += 1
            continue

        if not line.strip():
            i += 1
            continue

        # Series header
        if line.strip().startswith("──"):
            series_name = line.strip().strip("─").strip()
            envs[current_env]["series"][series_name] = []
            i += 1
            while i < len(lines) and lines[i].strip().startswith("↳"):
                sub_line = lines[i].strip()
                parts = re.split(r"\s{2,}", sub_line)
                name = parts[0]
                vals = parts[1:]
                row = {}
                for h, v in zip(headers, vals):
                    val = v.split("±")[0]
                    row[h] = float(val) if val != "-" else np.nan
                envs[current_env]["series"][series_name].append((name, row))
                i += 1
            continue

        # Single point
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) > 1:
            name = parts[0]
            vals = parts[1:]
            row = {}
            for h, v in zip(headers, vals):
                val = v.split("±")[0]
                row[h] = float(val) if val != "-" else np.nan
            envs[current_env]["points"][name] = row

        i += 1

    return envs


def get_x_val(name, row, pjsvd_x):
    if "PJSVD" in name:
        if pjsvd_x == "corrected":
            return row.get("Corr-L2-ID", np.nan)
        else:
            return row.get("Uncorr-L2-ID", np.nan)
    else:
        return row.get("Uncorr-L2-ID", np.nan)


def plot_metrics(envs, pjsvd_x):
    out_dir = f"plots_l2_{pjsvd_x}"
    os.makedirs(out_dir, exist_ok=True)

    for env, data in envs.items():
        base_x_col = "Uncorr-L2-ID"

        all_metrics = []
        if data["points"]:
            all_metrics = list(list(data["points"].values())[0].keys())
        elif data["series"]:
            all_metrics = list(list(data["series"].values())[0][0][1].keys())

        # Plot all metrics against x
        y_cols = [
            m
            for m in all_metrics
            if m
            not in [
                "Uncorr-L2-ID",
                "Corr-L2-ID",
                "Uncorr-L2-OOD",
                "Corr-L2-OOD",
                "Train (s)",
                "Eval (s)",
            ]
        ]
        y_cols.extend(["Uncorr-L2-OOD", "Corr-L2-OOD"])

        for y_col in y_cols:
            plt.figure(figsize=(9, 6))

            # Plot series
            for series_name, rows in data["series"].items():
                xs = []
                ys = []
                for name, row in rows:
                    x_val = get_x_val(series_name, row, pjsvd_x)
                    if not np.isnan(x_val) and not np.isnan(row.get(y_col, np.nan)):
                        xs.append(x_val)
                        ys.append(row[y_col])

                if xs and ys:
                    pts = sorted(zip(xs, ys))
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    plt.plot(xs, ys, marker="o", label=series_name)

            # Plot single points and baselines
            colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
            c_idx = len(data["series"]) % len(colors)

            for pt_name, row in data["points"].items():
                x_val = get_x_val(pt_name, row, pjsvd_x)
                y_val = row.get(y_col, np.nan)

                if np.isnan(x_val):
                    # Baseline (e.g. Deep Ensemble)
                    if not np.isnan(y_val) and (
                        "Ensemble" in pt_name or "Dropout" in pt_name
                    ):
                        plt.axhline(
                            y_val,
                            label=f"{pt_name} (Baseline)",
                            linestyle="--",
                            color=colors[c_idx],
                        )
                        c_idx = (c_idx + 1) % len(colors)
                else:
                    if not np.isnan(y_val):
                        plt.scatter(
                            [x_val],
                            [y_val],
                            label=pt_name,
                            zorder=5,
                            s=150,
                            marker="X",
                            color=colors[c_idx],
                        )
                        c_idx = (c_idx + 1) % len(colors)

            # Use log scale appropriately if values span several magnitudes
            plt.xscale("log")

            plt.xlabel(f"Perturbation L2 ({pjsvd_x} for PJSVD)")
            plt.ylabel(y_col)
            plt.title(f"{env}: {y_col} vs Perturbation L2")
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            safe_env = env.replace(" ", "_")
            safe_y = y_col.replace(" ", "_").replace("/", "_")
            out_path = os.path.join(out_dir, f"{safe_env}_{safe_y}_vs_{pjsvd_x}.png")
            plt.savefig(out_path)
            plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot metrics vs Perturbation L2")
    parser.add_argument(
        "--pjsvd-x",
        choices=["uncorrected", "corrected"],
        default="uncorrected",
        help="Which L2 metric to use for PJSVD on the x-axis",
    )
    args = parser.parse_args()

    envs = parse_report("report.txt")
    plot_metrics(envs, args.pjsvd_x)
    print(f"Plots successfully saved to 'plots_l2_{args.pjsvd_x}' directory.")
