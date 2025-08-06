from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from tqdm import tqdm
from pNK_structured_MC import Paralleler, Statistician  # Import your structured classes

# === Define Sobol Problem ===
problem = {
    'num_vars': 8,
    'names': ['alpha', 'K', 'chi_ms', 'chi_markup', 'w_init', 'chi_w', 'iota', 'iota_long_ratio'],
    'bounds': [
        [0.5, 1.5],      # alpha
        [2, 11],         # K
        [0.1, 2.0],      # chi_ms
        [0.1, 2.0],      # chi_markup
        [0.0, 5.0],      # w_init
        [0.01, 0.5],     # chi_w
        [0.5, 2.0],      # iota
        [0.5, 1.0]       # iota_long_ratio
    ]
}

# === Sobol Sampling ===
n_samples = 128  # Base sample size
param_values = saltelli.sample(problem, n_samples, calc_second_order=False)
sobol_df = pd.DataFrame(param_values, columns=problem['names'])
sobol_df['K'] = sobol_df['K'].round().astype(int)

# === Fixed parameters ===
fixed_params = {
    'T_max': 600, 'N': 15, 'h': 6, 'J': 20, 'M': 20,
    'initial_markup': 1.0, 'price_rule': 0, 'procyclical': 0,
    'free_research': False
}

# === Outputs storage ===
Y_productivity, Y_sustainability, Y_HHI = [], [], []

# === Loop over Sobol samples ===
available_cores = max(1, multiprocessing.cpu_count() - 1)

for idx, row in tqdm(sobol_df.iterrows(), total=len(sobol_df), desc="Sobol Analysis"):
    # Build parameter set
    params = fixed_params.copy()
    params.update({
        'alpha': row['alpha'],
        'K': int(row['K']),
        'chi_ms': row['chi_ms'],
        'chi_markup': row['chi_markup'],
        'w_init': row['w_init'],
        'w_final': 0,  # Keep sustainability preference final as before
        'Tstart_w': 1, 'Tend_w': 1,
        'chi_w': row['chi_w'],
        'iota': row['iota'],
        'iota_long_ratio': row['iota_long_ratio']
    })

    # Run Monte Carlo simulations
    results, stats, metrics = Paralleler.run_mc_fixed(
        params,
        n_runs=20,                # MC simulations per Sobol sample
        n_processes=available_cores,
        overlay_debug=False,
        auto_plot=False,
        debug_print=False,
        variable_landscape=False  # Fix landscape for comparability
    )

    # Extract final outputs from summary table
    summary = stats.summary_table(metrics)
    Y_productivity.append(float(summary.loc[summary['Metric'] == 'productivity_final', 'Value']))
    Y_sustainability.append(float(summary.loc[summary['Metric'] == 'sustainability_final', 'Value']))
    Y_HHI.append(float(summary.loc[summary['Metric'] == 'market_share_final', 'Value']))  # Use HHI proxy

# === Attach outputs to dataframe ===
sobol_df['productivity'] = Y_productivity
sobol_df['sustainability'] = Y_sustainability
sobol_df['HHI'] = Y_HHI

# === Sobol Analysis ===
Si_prod = sobol.analyze(problem, sobol_df['productivity'].values, print_to_console=True, calc_second_order=False)
Si_sust = sobol.analyze(problem, sobol_df['sustainability'].values, print_to_console=True, calc_second_order=False)
Si_HHI  = sobol.analyze(problem, sobol_df['HHI'].values, print_to_console=True, calc_second_order=False)

# === Helper: Format Sobol indices ===
def format_sobol_indices(Si, output_name):
    df = pd.DataFrame({
        'S1': Si['S1'],
        'ST': Si['ST'],
        'S1_conf': Si['S1_conf'],
        'ST_conf': Si['ST_conf']
    }, index=problem['names'])
    df['Output'] = output_name
    return df.set_index('Output', append=True).reorder_levels(['Output', df.index.name])

sobol_all = pd.concat([
    format_sobol_indices(Si_prod, 'productivity'),
    format_sobol_indices(Si_sust, 'sustainability'),
    format_sobol_indices(Si_HHI, 'HHI')
])

# === Plot Sobol Results ===
def plot_sobol_results(sobol_all, figsize=(12, 8)):
    outputs = sobol_all.index.get_level_values('Output').unique()
    params = sobol_all.index.get_level_values(1).unique()
    fig, axes = plt.subplots(nrows=len(outputs), ncols=2, figsize=figsize, sharey=True, constrained_layout=True)

    colors = {'productivity': 'blue', 'sustainability': 'green', 'HHI': 'purple'}
    titles = {'S1': 'First-order Sobol index', 'ST': 'Total-order Sobol index'}

    for i, output in enumerate(outputs):
        data = sobol_all.xs(output, level='Output')
        for j, index_type in enumerate(['S1', 'ST']):
            ax = axes[i, j]
            ax.errorbar(data[index_type], params,
                       xerr=data[f"{index_type}_conf"], fmt='o',
                       color=colors[output], ecolor='gray', capsize=5)
            ax.set_xlim(left=0)
            if j == 0: ax.set_ylabel(output, fontsize=12)
            if i == 0: ax.set_title(titles[index_type], fontsize=12)
            if j != 0: ax.set_yticklabels([])
    fig.suptitle('Sobol Sensitivity Analysis Results', fontsize=14, y=1.02)
    for ax in axes[:, 0]:
        ax.set_yticks(range(len(params)))
        ax.set_yticklabels(params)
    return fig, axes

plot_sobol_results(sobol_all)
plt.show()
# === Optional CSV exports ===
sobol_df.to_csv("sobol_samples_outputs.csv", index=False)
sobol_all.reset_index().to_csv("sobol_indices.csv", index=False)