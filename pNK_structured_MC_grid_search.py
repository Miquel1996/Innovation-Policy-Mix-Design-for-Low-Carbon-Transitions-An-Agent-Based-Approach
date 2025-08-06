import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
from functools import partial
import itertools
from pNK_structured_MC import Paralleler, Statistician, debug_params

# === Smart core allocator ===
def compute_core_allocation(total_cores, n_param_sets, target_mc_runs=20, min_inner=2, max_inner=12):
    """
    Dynamically split available cores between outer (param sets) and inner (MC runs).
    """
    outer = min(n_param_sets, total_cores // min_inner)
    inner = max(min(total_cores // outer, max_inner), min_inner)
    return outer, inner

# === Worker function for each parameter set ===
def run_param_combo(combo_dict, base_params, n_runs, n_inner_cores):
    params = base_params.copy()
    params.update(combo_dict)

    results, stats, metrics = Paralleler.run_mc_fixed(
        params,
        n_runs=n_runs,
        n_processes=n_inner_cores,
        overlay_debug=False,
        auto_plot=False,
        debug_print=False,
        variable_landscape=False,
        outer_desc=f"MC for {combo_dict}"
    )


    df_summary = stats.summary_table(metrics).set_index("Metric")["Value"]

    # Extract metrics
    skew_prod = metrics['productivity']['skew']['median'][-1]
    std_prod = metrics['productivity']['std']['median'][-1]
    skew_ms = metrics['market_share']['skew']['median'][-1]
    hhi = metrics['market_share']['mv']['median'][-1]
    markup = metrics['markup']['mv']['median'][-1]
    std_markup = metrics['markup']['std']['median'][-1]
    skew_markup = metrics['markup']['skew']['median'][-1]
    avg_prod_growth = df_summary["avg_productivity_growth"]
    avg_rd_success = df_summary["avg_R&D_success"]

    # === Compute score ===
    score = 0
    if skew_prod > 0: score += 1/9
    if std_prod >= 0.05: score += 1/9
    if skew_ms > 0: score += 1/9
    if 0.15 <= hhi <= 0.35: score += 1/9
    if 1.0 < markup < 1.5: score += 1/9
    if 0.20 <= std_markup <= 0.40: score += 1/9
    if 0.10 <= skew_markup: score += 1/9
    if 0.0005 <= avg_prod_growth <= 0.0023: score += 1/9
    if 0.02 <= avg_rd_success <= 0.05: score += 1/9

    return {
        **combo_dict,
        "score": score,
        "skew_prod": skew_prod,
        "std_prod": std_prod,
        "skew_market_share": skew_ms,
        "hhi": hhi,
        "markup": markup,
        "std_markup": std_markup,
        "skew_markup": skew_markup,
        "avg_productivity_growth": avg_prod_growth,
        "avg_R&D_success": avg_rd_success
    }

# === Grid search function ===
def grid_search(base_params, param_grid, n_runs=20, output_csv="Validation_tables.csv"):
    param_names = list(param_grid.keys())
    all_combinations = [dict(zip(param_names, combo)) for combo in itertools.product(*param_grid.values())]

    total_cores = multiprocessing.cpu_count()
    outer_cores, inner_cores = compute_core_allocation(total_cores, n_param_sets=len(all_combinations))
    print(f"Detected {total_cores} cores → Using {outer_cores} outer × {inner_cores} inner = {outer_cores * inner_cores} total cores")

    func = partial(run_param_combo, base_params=base_params, n_runs=n_runs, n_inner_cores=inner_cores)

    # Parallel execution over parameter combinations
    with multiprocessing.Pool(outer_cores) as pool:
        score_data = list(tqdm(pool.imap(func, all_combinations), total=len(all_combinations), desc="Grid Search"))

    # Build DataFrame
    score_df = pd.DataFrame(score_data)
    score_df.to_csv(output_csv, index=False)
    print(f"✅ Results saved to {output_csv}")

    # Heatmap for 2D grids
    if len(param_names) == 2:
        plt.figure(figsize=(8, 6))
        pivot_table = score_df.pivot(index=param_names[0], columns=param_names[1], values='score')
        sns.heatmap(pivot_table.sort_index(ascending=True), cmap='viridis', annot=True, cbar_kws={'label': 'Score'})
        plt.title(f"Validation Score across {param_names[0]} and {param_names[1]}")
        plt.xlabel(param_names[1])
        plt.ylabel(param_names[0])
        plt.tight_layout()
        plt.show()

    return score_df

# === Run grid search ===
if __name__ == "__main__":
    param_grid = {
        'K': [5, 6, 7],
        'chi_markup': [0.2, 0.225, 0.25],
        'iota': [1, 1.25, 1.5]  # Example of adding w_final exploration
    }

    score_df = grid_search(
        base_params=debug_params,
        param_grid=param_grid,
        n_runs=20,
        output_csv="Validation_tables.csv"
    )
