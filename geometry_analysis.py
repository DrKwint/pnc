import os
import glob
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, linregress
from typing import Dict, Any


def analyze_geometry_metrics(
    npz_path: str, p_size: float, env_name: str, out_dir: str = "geometry_analysis", no_plots: bool = False
) -> Dict[str, Any]:
    """
    Reads the geometry metrics NPZ file and computes correlation, regression,
    and bucketing, saving figures to out_dir.
    """
    data = np.load(npz_path)

    geom_dist_id = data["geom_dist_id"].flatten()
    geom_dist_ood = data["geom_dist_ood"].flatten()

    corr_l2_id = data["corr_l2_id"].flatten()
    corr_l2_ood = data["corr_l2_ood"].flatten()

    uncorr_l2_id = data["uncorr_l2_id"].flatten()
    uncorr_l2_ood = data["uncorr_l2_ood"].flatten()

    os.makedirs(out_dir, exist_ok=True)

    # 1. Compute correlations
    results = {}
    for split, geom, corr in [
        ("ID", geom_dist_id, corr_l2_id),
        ("OOD", geom_dist_ood, corr_l2_ood),
    ]:
        # Pearson
        pearson_r, _ = pearsonr(geom, corr)
        # Spearman
        spearman_r, _ = spearmanr(geom, corr)
        # Linear Regression
        slope, intercept, r_value, p_value, std_err = linregress(geom, corr)

        results[split] = {
            "pearson": pearson_r,
            "spearman": spearman_r,
            "slope": slope,
            "intercept": intercept,
            "r_squared": r_value**2,
        }

    # Combined
    all_geom = np.concatenate([geom_dist_id, geom_dist_ood])
    all_corr = np.concatenate([corr_l2_id, corr_l2_ood])
    p_all, _ = pearsonr(all_geom, all_corr)
    s_all, _ = spearmanr(all_geom, all_corr)
    slope_all, intercept_all, r_val_all, _, _ = linregress(all_geom, all_corr)

    results["Combined"] = {
        "pearson": p_all,
        "spearman": s_all,
        "slope": slope_all,
        "intercept": intercept_all,
        "r_squared": r_val_all**2,
    }

    # 2. Bucketing
    def bucket_data(geom, corr, n_buckets=10):
        # Quantile-based bucketing
        quantiles = np.linspace(0, 1, n_buckets + 1)
        bins = np.quantile(geom, quantiles)
        bins[-1] += 1e-5  # include max

        bucket_centers = []
        bucket_means = []
        bucket_medians = []

        for i in range(n_buckets):
            mask = (geom >= bins[i]) & (geom < bins[i + 1])
            if np.sum(mask) > 0:
                bucket_centers.append(np.mean(geom[mask]))
                bucket_means.append(np.mean(corr[mask]))
                bucket_medians.append(np.median(corr[mask]))

        return bucket_centers, bucket_means, bucket_medians

    bc_id, bm_id, bmed_id = bucket_data(geom_dist_id, corr_l2_id)
    bc_ood, bm_ood, bmed_ood = bucket_data(geom_dist_ood, corr_l2_ood)

    results["Buckets_ID"] = {"centers": bc_id, "means": bm_id, "medians": bmed_id}
    results["Buckets_OOD"] = {"centers": bc_ood, "means": bm_ood, "medians": bmed_ood}

    # 3. Plotting
    if not no_plots:
        base_name = os.path.basename(npz_path).replace(".npz", "")

        # Scatter plot
        plt.figure(figsize=(8, 6))
        plt.scatter(geom_dist_id, corr_l2_id, alpha=0.1, color="blue", label="ID")
        plt.scatter(geom_dist_ood, corr_l2_ood, alpha=0.1, color="red", label="OOD")

        # Trendlines
        x_id = np.linspace(geom_dist_id.min(), geom_dist_id.max(), 100)
        plt.plot(
            x_id,
            results["ID"]["intercept"] + results["ID"]["slope"] * x_id,
            color="darkblue",
            lw=2,
        )

        x_ood = np.linspace(geom_dist_ood.min(), geom_dist_ood.max(), 100)
        plt.plot(
            x_ood,
            results["OOD"]["intercept"] + results["OOD"]["slope"] * x_ood,
            color="darkred",
            lw=2,
        )

        plt.xlabel("Geometry Distance $d(x)$")
        plt.ylabel("Post-Correction Residual (Corr-L2)")
        plt.title(f"{env_name}: Corr-L2 vs Geometry (Perturbation={p_size})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{base_name}_scatter.png"), dpi=150)
        plt.close()

        # Bucketed plot
        plt.figure(figsize=(8, 6))
        plt.plot(bc_id, bm_id, "o-", color="blue", label="ID Mean")
        plt.plot(bc_ood, bm_ood, "s-", color="red", label="OOD Mean")
        plt.plot(bc_id, bmed_id, "--", color="lightblue", label="ID Median")
        plt.plot(bc_ood, bmed_ood, "--", color="lightcoral", label="OOD Median")

        plt.xlabel("Geometry Distance $d(x)$")
        plt.ylabel("Post-Correction Residual (Corr-L2)")
        plt.title(f"{env_name}: Bucketed Means & Medians (Perturbation={p_size})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{base_name}_buckets.png"), dpi=150)
        plt.close()

    return results


def print_report(results: Dict[str, Any], npz_path: str):
    print(f"=== Geometry Analysis Report for {os.path.basename(npz_path)} ===")
    for split in ["ID", "OOD", "Combined"]:
        res = results[split]
        print(
            f"[{split}] Pearson R: {res['pearson']:.4f} | Spearman R: {res['spearman']:.4f}"
        )
        print(
            f"[{split}] Regr: slope = {res['slope']:.4f}, R^2 = {res['r_squared']:.4f}"
        )
    print("=" * 60)


def extract_info_from_path(path: str):
    """
    Tries to extract env_name and p_size from the path.
    Example path: results/Hopper-v5/some_experiment_ps32_geometry.npz
    """
    parts = path.split(os.sep)
    env_name = "Unknown"
    for part in parts:
        if part in ["Hopper-v5", "Walker2d-v5", "HalfCheetah-v5", "Ant-v5", "Pusher-v5", "Reacher-v5"]:
            env_name = part
            break
        elif "results" not in part and part != "" and os.path.isdir(os.path.join(*parts[:parts.index(part)+1])):
            # Heuristic for env name if not in known list
             if part != "results":
                env_name = part

    # Try to find ps[NUMBER] in filename
    filename = os.path.basename(path)
    p_size = 20.0  # default
    if "_ps" in filename:
        try:
            p_str = filename.split("_ps")[-1].split("_")[0].replace("geometry.npz", "").replace(".npz", "")
            p_size = float(p_str)
        except ValueError:
            pass

    return env_name, p_size


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze geometry distance vs corr-l2")
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default="results",
        help="Path to *_geometry.npz file, directory, or glob pattern",
    )
    parser.add_argument(
        "--p_size",
        type=float,
        default=None,
        help="Override perturbation size (for plot titling)",
    )
    parser.add_argument("--env", type=str, default=None, help="Override environment name")
    parser.add_argument(
        "--out",
        type=str,
        default="geometry_analysis",
        help="Output directory for plots",
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--no_plots", action="store_true", help="Skip individual plotting")

    args = parser.parse_args()

    # Find files
    if os.path.isfile(args.input):
        files = [args.input]
    elif os.path.isdir(args.input):
        files = glob.glob(os.path.join(args.input, "**/*_geometry.npz"), recursive=True)
    else:
        files = glob.glob(args.input, recursive=True)

    if not files:
        print(f"No geometry files found for input: {args.input}")
        exit(0)

    print(f"Found {len(files)} geometry files to process.")

    def process_file(f):
        env_name, p_size = extract_info_from_path(f)
        if args.env:
            env_name = args.env
        if args.p_size is not None:
            p_size = args.p_size

        try:
            res = analyze_geometry_metrics(f, p_size, env_name, args.out, no_plots=args.no_plots)
            return (f, env_name, p_size, res)
        except Exception as e:
            print(f"Error processing {f}: {e}")
            return None

    all_results = []
    if len(files) == 1 or args.workers == 1:
        for f in sorted(files):
            print(f"Processing {f}...")
            res = process_file(f)
            if res:
                all_results.append(res)
                print_report(res[3], f)
    else:
        print(f"Using {args.workers} workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_file = {executor.submit(process_file, f): f for f in files}
            for future in concurrent.futures.as_completed(future_to_file):
                res = future.result()
                if res:
                    all_results.append(res)
                    print(f"Done: {os.path.basename(res[0])}")

    # Summary report
    if len(all_results) > 1:
        all_results.sort(key=lambda x: x[0])  # sort by path
        summary_path = os.path.join(args.out, "summary_report.txt")
        with open(summary_path, "w") as sf:
            sf.write("=== Batch Geometry Analysis Summary ===\n\n")
            sf.write(f"{'File':<60} | {'Env':<15} | {'P-Size':<8} | {'ID Pearson':<10} | {'OOD Pearson':<10}\n")
            sf.write("-" * 115 + "\n")
            for f, env, ps, res in all_results:
                fname = os.path.basename(f)
                id_p = res["ID"]["pearson"]
                ood_p = res["OOD"]["pearson"]
                sf.write(f"{fname[:60]:<60} | {env:<15} | {ps:<8.1f} | {id_p:<10.4f} | {ood_p:<10.4f}\n")

        print(f"\nSummary report saved to {summary_path}")
