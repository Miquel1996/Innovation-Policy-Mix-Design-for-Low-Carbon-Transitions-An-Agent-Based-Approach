import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing
from pNK_structured_MC import Paralleler, Statistician, debug_params  # ✅ import model code

def grid_search(base_params, param_grid, n_runs=5, n_processes=None, ci=95, output_csv="Validation_tables.csv"):
    """
    Performs grid search over a restricted parameter space for the ABM model.
    """
    if n_processes is None:
        n_processes = max(1, multiprocessing.cpu_count() - 1)

    param_names = list(param_grid.keys())
    all_combinations = list(itertools.product(*param_grid.values()))

    results_dict = {}
    score_data = []

    for combo in tqdm(all_combinations, desc="Grid Search"):
        # Update parameters for this combination
        params = base_params.copy()
        for name, value in zip(param_names, combo):
            params[name] = value

        # Run MC simulations
        mc_results, stats, metrics = Paralleler.run_mc_fixed(
            params,
            n_runs=n_runs,
            n_processes=n_processes,
            overlay_debug=False,
            auto_plot=False,
            debug_print=False,
            variable_landscape=False
        )

        # Summarize metrics
        df_summary = stats.summary_table(metrics).set_index("Metric")["Value"]

        # Extract relevant metrics
        skew_prod = metrics['productivity']['skew']['median'][-1]
        std_prod = metrics['productivity']['std']['median'][-1]
        skew_ms = metrics['market_share']['skew']['median'][-1]
        hhi = metrics['market_share']['mv']['median'][-1]
        markup = metrics['markup']['mv']['median'][-1]
        std_markup = metrics['markup']['std']['median'][-1]
        skew_markup = metrics['markup']['skew']['median'][-1]
        avg_productivity_growth = df_summary["avg_productivity_growth"]
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
        if 0.0005 <= avg_productivity_growth <= 0.0023: score += 1/9  # (converted %)
        if 0.02 <= avg_rd_success <= 0.05: score += 1/9

        # Store results
        key = tuple(combo)
        results_dict[key] = metrics
        score_data.append({**{p: v for p, v in zip(param_names, combo)}, "score": score})

    # === Build results DataFrame ===
    score_df = pd.DataFrame(score_data)
    score_df.to_csv(output_csv, index=False)
    print(f"✅ Results saved to {output_csv}")

    # === Optional: Heatmap for 2D parameter grid ===
    if len(param_names) == 2:
        plt.figure(figsize=(8, 6))
        pivot_table = score_df.pivot(index=param_names[0], columns=param_names[1], values='score')
        sns.heatmap(pivot_table.sort_index(ascending=True), cmap='viridis', annot=True, cbar_kws={'label': 'Score'})
        plt.title(f"Validation Score across {param_names[0]} and {param_names[1]}")
        plt.xlabel(param_names[1])
        plt.ylabel(param_names[0])
        plt.tight_layout()
        plt.show()

    return score_df, results_dict


# === RUN IF FILE EXECUTED DIRECTLY ===
if __name__ == "__main__":
    import itertools  # ✅ Needed for param combinations
    
    # Example restricted grid
    param_grid = {
        'K': [4, 6],
        'chi_markup': [0.2, 0.25]
    }

    score_df, results = grid_search(
        base_params=debug_params,
        param_grid=param_grid,
        n_runs=5,
        n_processes=4,
        output_csv="Validation_tables.csv"
    )
