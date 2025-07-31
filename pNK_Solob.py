from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import time
from matplotlib.lines import Line2D  # For custom legend
from scipy.stats import skew
from tqdm import tqdm

# Start timer
start_time = time.time()

# Step 1: Define the model input space (parameters)
problem = {
    'num_vars': 8,
    'names': [
        'alpha',
        'K',
        'chi_ms',
        'chi_markup',
        'w',
        'chi_w',
        'iota',
        'iota_long_ratio'
    ],
    'bounds': [
        [0.5, 1.5],      # alpha
        [2, 11],         # K (discrete, but will round later)
        [0.1, 2.0],     # chi_ms
        [0.1, 2.0],     # chi_markup
        [0.0, 5.0],      # w
        [0.01, 0.5],     # chi_w
        [0.5, 2.0],      # iota
        [0.5, 1]       # iota_long_ratio
    ]
}

# Step 2: Generate Sobol sequences (sample parameter combinations)
n_samples = 124  # base sample size, total runs = n_samples * (2D + 2)

# Generate samples first
param_values = saltelli.sample(problem, n_samples, calc_second_order=False)

# Convert to DataFrame
sobol_df = pd.DataFrame(param_values, columns=problem['names'])
sobol_df['K'] = sobol_df['K'].round().astype(int)

# Initialize progress bar
pbar = tqdm(total=len(sobol_df), desc="Overall progress", position=0)

# Other fixed parameters
n_simulations = 20 # Number of simulations per parameter set
T_max = 600 # Number of time-steps epr simulation
N = 15 # Number of components (NK landscape)
h = 6 # Correlation degree (NK landscape) (h=N : -1 correl; h=0 : 1 correl)
J = 20 # Number of firms
M = 20 # Firm memory size
initial_markup = 1 # Markup of all firms at t=0
price_rule = 0 # Price rule: 0 == all penetration, 1 == all skiming, 2== mixed
procyclical = 0 # Variable governing markup/price changes: 0 == market share, 1== profit
free_research = False # Is research free?

# SET SEEDS
research_seeds = np.arange(1, n_simulations+1)
seed_landscape = 1
rng_landscape = np.random.RandomState(seed=seed_landscape)

#DETERMINE LANDSCAPE GLOBAL MAXIMA
X1_star = rng_landscape.randint(0, 2, N)
flip_indices = rng_landscape.choice(N, size=h, replace=False)
X2_star = X1_star.copy()
X2_star[flip_indices] = 1 - X2_star[flip_indices] 

# Define the outputs you care about
Y_productivity = []
Y_HHI = []
Y_sustainability = []

for i, row in sobol_df.iterrows():
    # Unpack parameters
    alpha = row['alpha']
    K = int(row['K'])  # Ensure K is an integer
    chi_ms = row['chi_ms']
    chi_markup = row['chi_markup']
    w = row['w']
    chi_w = row['chi_w']
    iota = row['iota']
    iota_long_ratio = row['iota_long_ratio']
    iota_long = iota * iota_long_ratio  # Ensures iota_long ≤ iota
    w_init = w  # Initial weight to sustainability (firms)
    chi_p = chi_markup * initial_markup / (1 + initial_markup)
    
    # Optional: Print progress
    print(f"Running simulation for alpha={alpha:.2f}, K={K}, chi_ms={chi_ms:.2f}, chi_markup={chi_markup:.2f}, w={w:.2f}, chi_w={chi_w:.2f}, iota={iota:.2f}, iota_long={iota_long:.2f}")
    # Update progress bar description with current parameters
    pbar.set_description(f"Running: α={alpha:.2f}, K={K}, χ_ms={chi_ms:.2f}, χ_mu={chi_markup:.2f}")
    # Initialize storage for this parameter combination
    all_weighted_productivity = np.zeros((n_simulations, T_max))
    all_weighted_sustainability = np.zeros((n_simulations, T_max))
    all_HHI_series = np.zeros((n_simulations, T_max))

    #CREATE pNK LANDSCAPE
    def hamming_distance(x, y): 
        return np.sum(x != y) 

    def neighbors(x, dist=1):
        candidates = []
        for indices in itertools.combinations(range(N), dist):
            x_new = x.copy()
            x_new[list(indices)] = 1 - x_new[list(indices)]
            candidates.append(x_new)
        return np.array(candidates)

    def build_interaction_matrix(N, K, alpha):
        A = np.zeros((N, N))
        for i in range(N):
            is_even = (i % 2 == 0)
            upper_thresh = i + K // 2 + (0.5 if i % 2 == 1 else 0)
            lower_thresh = i - K // 2 + (-0.5 if is_even else 0)
            for j in range(N):
                if j == i:
                    A[i, j] = 1
                elif j > i:
                    if j <= upper_thresh:
                        A[i, j] = alpha
                    elif j >= N + lower_thresh:
                        A[i, j] = -alpha
                else:
                    if j >= lower_thresh:
                        A[i, j] = -alpha
                    elif j <= upper_thresh - N:
                        A[i, j] = alpha
        return A

    def compute_c_vector(X_star, A):
        return np.array([X_star[i] + np.dot(A[i], X_star) for i in range(N)])

    def phi_i(i, X, A, c_vector):
        interaction_sum = np.dot(A[i], X)
        return 1 / (1 + abs(X[i] + interaction_sum - c_vector[i]))

    def fitness(X, A, c_vector):
        return np.mean([phi_i(i, X, A, c_vector) for i in range(N)])

    # --- INITIALIZE LANDSCAPES ---
    A = build_interaction_matrix(N, K, alpha)
    c1 = compute_c_vector(X1_star, A)
    c2 = compute_c_vector(X2_star, A)

    # Run Monte Carlo simulations
    for sim in (range(n_simulations)):
        seed_research = research_seeds[sim]
        rng_research = np.random.RandomState(seed=seed_research)
        
        # Initialize economic variables
        market_shares = np.ones(J) / J
        market_share_history = np.zeros((J, T_max))
        profit_history = np.zeros((J, T_max))
        profit_rate_history = np.zeros((J,T_max))
        price_history = np.zeros((J, T_max))
        fitness_history = np.zeros((J, T_max))
        markup_history = np.zeros((J, T_max))
        markup_history[:, 0] = initial_markup
        firms = []
        f1_history = np.zeros((J, T_max))
        f2_history = np.zeros((J, T_max))
        w_history = np.zeros((J, T_max))
        units_history = np.zeros((J, T_max))
        impact_history = np.zeros((J, T_max))

        for j in range(J):
            x0 = 1 - X1_star
            memory = [x0.copy()]
            timers = [0]
            firms.append({
                'x': x0,
                'w': w_init,
                'memory': memory,
                'timers': timers
            })
            w_history[j, 0] = w

        # --- ECONOMIC INTERACTIONS
        for t in range(T_max):
            # First update firm positions based on R&D
            for j in range(J):
                firm = firms[j]
                x_current = firm['x']
                memory = firm['memory']
                
                if t > 0:
                    search_prob = 1 - np.exp(-iota * np.clip(profit_history[j, t-1], 0, 1))
                    gets_to_search = rng_research.rand() < search_prob
                else:
                    gets_to_search = True
                
                if gets_to_search or free_research:
                    found = False
                    for d in [1, 2]:
                        candidates = neighbors(x_current, dist=d)
                        candidates = [x for x in candidates if not any(np.array_equal(x, m) for m in memory)]
                        if candidates:
                            if d == 2:
                                long_search_prob = 1 - np.exp(iota_long * np.clip(profit_history[j, t-1] if t > 0 else 1, 0, 1))
                                gets_long_jump = rng_research.rand() < long_search_prob
                                if gets_long_jump or free_research:
                                    x_explore = candidates[rng_research.randint(len(candidates))]
                                    found = True
                                    break
                                else:
                                    continue
                            else:
                                x_explore = candidates[rng_research.randint(len(candidates))]
                                found = True
                                break
                    if not found:
                        x_explore = x_current.copy()

                    memory.append(x_explore.copy())
                    firm['timers'].append(0)

                    f1 = fitness(x_explore, A, c1)
                    f2 = fitness(x_explore, A, c2)
                    eval_explore = f1 * (1 + firm['w'] * f2)

                    f1_old = fitness(x_current, A, c1)
                    f2_old = fitness(x_current, A, c2)
                    eval_current = f1_old * (1 + firm['w'] * f2_old)

                    if eval_explore >= eval_current:
                        firm['x'] = x_explore.copy()

                for i in range(len(memory)):
                    if np.array_equal(memory[i], firm['x']):
                        firm['timers'][i] = 0
                    else:
                        firm['timers'][i] += 1

                if len(memory) > M:
                    current_tech = firm['x']
                    neighbor_techs = neighbors(current_tech, dist=1)
                    non_neighbor_indices = [
                        i for i in range(len(memory)) 
                        if not any(np.array_equal(memory[i], neighbor) for neighbor in neighbor_techs)
                    ]
                    if non_neighbor_indices:
                        timers_non_neighbors = [firm['timers'][i] for i in non_neighbor_indices]
                        idx_to_remove = non_neighbor_indices[np.argmax(timers_non_neighbors)]
                        memory.pop(idx_to_remove)
                        firm['timers'].pop(idx_to_remove)
                    else:
                        idx_remove = np.argmax(firm['timers'])
                        memory.pop(idx_remove)
                        firm['timers'].pop(idx_remove)

                f1_history[j, t] = fitness(firm['x'], A, c1)
                f2_history[j, t] = fitness(firm['x'], A, c2)
            
            # Then calculate economic variables
            f1_values = np.array([fitness(firms[j]['x'], A, c1) for j in range(J)])
            f2_values = np.array([fitness(firms[j]['x'], A, c2) for j in range(J)])
            
            if t > 1:
                if procyclical == 0:
                    changes = (market_share_history[:, t-1] - market_share_history[:, t-2]) / (market_share_history[:, t-2] + 1e-10)
                else:
                    changes = (profit_history[:, t-1] - profit_history[:, t-2]) / (profit_history[:, t-2] + 1e-10)
            else:
                changes = np.zeros(J)
            
            current_market_shares = market_share_history[:, t-1] if t > 0 else np.ones(J)/J
            
            for j in range(J):
                if t > 0:
                    if price_rule == 1 or (current_market_shares[j] > (1/J) and price_rule == 2):
                        new_price = price_history[j, t-1] * (1 + chi_p * changes[j])
                        new_price = np.clip(new_price, 0, 50)
                        price_history[j, t] = new_price
                        markup_history[j, t] = price_history[j, t] * f1_values[j] - 1
                    else:
                        new_markup = markup_history[j, t-1] * (1 + chi_markup * changes[j])
                        new_markup = np.clip(new_markup, 0, 50)
                        markup_history[j, t] = new_markup
                        price_history[j, t] = (1 + markup_history[j, t]) / f1_values[j]
                else:
                    markup_history[j, t] = initial_markup
                    price_history[j, t] = (1 + initial_markup) / f1_values[j]
            
            current_prices = price_history[:, t]
            current_markups = markup_history[:, t]
            prices = (1 + current_markups) / f1_values
            firm_fitness = f1_values * (1 + w * f2_values) / (1 + current_markups)
            
            if t > 0:
                avg_fitness = np.sum(firm_fitness * market_share_history[:, t-1])
            else:
                avg_fitness = np.mean(firm_fitness)
            
            if t > 0:
                new_market_shares = market_share_history[:, t-1] * (1 + chi_ms * (firm_fitness / avg_fitness - 1))
            else:
                new_market_shares = market_shares * (1 + chi_ms * (firm_fitness / avg_fitness - 1))
            
            new_market_shares = np.clip(new_market_shares, 0.001, 0.999)
            new_market_shares = new_market_shares / np.sum(new_market_shares)
            
            units_produced = new_market_shares / prices
            impact = units_produced * (1 - f2_values)
            profits = units_produced * (current_markups / f1_values)
            profit_rate = profits / units_produced
            
            market_share_history[:, t] = new_market_shares
            profit_history[:, t] = profits
            profit_rate_history[:, t] = profit_rate
            price_history[:, t] = prices
            fitness_history[:, t] = firm_fitness
            units_history[:, t] = units_produced
            impact_history[:, t] = impact
            
            if t > 0:
                current_price = price_history[:, t]
                current_sustainability = f2_history[:, t]
                current_market_shares = market_share_history[:, t]
                w_j = firms[j]['w'] 
                
                for j in range(J):
                    competitors = [h for h in range(J) 
                                 if (current_market_shares[h] > current_market_shares[j])]
                    
                    valid_competitors = []
                    for h in competitors:
                        cond_b = not ((current_price[h] < current_price[j]) and 
                                     (current_sustainability[h] > current_sustainability[j]))
                        cond_c = not ((current_price[h] > current_price[j]) and 
                                     (current_sustainability[h] < current_sustainability[j]))
                        cond_d = (1 + w_j * current_sustainability[j]) / current_price[j] > (1 + w_j * current_sustainability[h]) / current_price[h]
                        
                        if cond_b and cond_c and cond_d:
                            valid_competitors.append(h)
                    
                    if valid_competitors:
                        h_star = valid_competitors[np.argmin(np.abs(current_price[valid_competitors] - current_price[j]))]
                        w_j = firms[j]['w']
                        w_h = firms[h_star]['w']
                        sustainability_ratio = current_sustainability[h_star] / (current_sustainability[j] + 1e-10)
                        new_w = w_j * (1 + chi_w * (sustainability_ratio - 1))
                        firms[j]['w'] = new_w
                    
                    w_history[j, t] = firms[j]['w']
            else:
                for j in range(J):
                    w_history[j, t] = firms[j]['w']

        # Calculate time series (totals)
        HHI_series = np.array([np.sum(market_share_history[:, t]**2) for t in range(T_max)])
        weights = units_history
        weighted_productivity = np.sum(f1_history * weights, axis=0) / np.sum(weights, axis=0)
        weighted_sustainability = np.sum(f2_history * weights, axis=0) / np.sum(weights, axis=0)
        
        # Store results from this simulation
        all_weighted_productivity[sim,:] = weighted_productivity
        all_weighted_sustainability[sim,:] = weighted_sustainability
        all_HHI_series[sim,:] = HHI_series
        
    # Store the median of the final values
    median_productivity = np.median(all_weighted_productivity[:, -1])  # final timestep
    median_sustainability = np.median(all_weighted_sustainability[:, -1])
    median_HHI = np.median(all_HHI_series[:, -1])
    
    #Append
    Y_productivity.append(median_productivity)
    Y_sustainability.append(median_sustainability)
    Y_HHI.append(median_HHI)
    # Update progress bar
    pbar.update(1)

sobol_df['productivity'] = Y_productivity
sobol_df['sustainability'] = Y_sustainability
sobol_df['HHI'] = Y_HHI

Si_prod = sobol.analyze(problem, sobol_df['productivity'].values, print_to_console=True, calc_second_order=False)
Si_sust = sobol.analyze(problem, sobol_df['sustainability'].values, print_to_console=True, calc_second_order=False)
Si_HHI  = sobol.analyze(problem, sobol_df['HHI'].values, print_to_console=True, calc_second_order=False)

# Create a helper function to reshape each Sobol index dictionary
def format_sobol_indices(Si, output_name):
    df = pd.DataFrame({
        'S1': Si['S1'],
        'ST': Si['ST'],
        'S1_conf': Si['S1_conf'],
        'ST_conf': Si['ST_conf']
    }, index=problem['names'])
    df['Output'] = output_name
    return df.set_index('Output', append=True).reorder_levels(['Output', df.index.name])

# Format each set of Sobol indices
sobol_prod = format_sobol_indices(Si_prod, 'productivity')
sobol_sust = format_sobol_indices(Si_sust, 'sustainability')
sobol_HHI  = format_sobol_indices(Si_HHI, 'HHI')

# Concatenate into a single DataFrame
sobol_all = pd.concat([sobol_prod, sobol_sust, sobol_HHI])

# Export to a single CSV
csv_path_sobol = r"C:\Users\mbassartilore\Documents\Recherche\EPOC\Projects\Mixing innovation policies\Sobol\sobol_indices.csv"
sobol_all.to_csv(csv_path_sobol)

def plot_sobol_results(sobol_all, figsize=(12, 8)):
    """
    Plot Sobol sensitivity indices (S1 and ST) for multiple outputs with error bars.
    
    Parameters:
    - sobol_all: DataFrame containing Sobol indices
    - figsize: Tuple for figure size
    """
    # Get unique outputs and parameter names
    outputs = sobol_all.index.get_level_values('Output').unique()
    params = sobol_all.index.get_level_values(1).unique()  # Get second level values
    
    # Create figure
    fig, axes = plt.subplots(nrows=len(outputs), ncols=2, 
                           figsize=figsize, 
                           sharey=True, 
                           constrained_layout=True)
    
    # Set colors for each output
    colors = {
        'productivity': 'blue',
        'sustainability': 'green',
        'HHI': 'purple'
    }
    
    titles = {
        'S1': 'First-order Sobol index',
        'ST': 'Total-order Sobol index'
    }
    
    # Plot each output
    for i, output in enumerate(outputs):
        # Get data for this output
        data = sobol_all.xs(output, level='Output')
        
        # Plot S1 and ST in separate columns
        for j, index_type in enumerate(['S1', 'ST']):
            ax = axes[i,j] if len(outputs) > 1 else axes[j]
            
            # Plot values with error bars
            ax.errorbar(data[index_type], params,
                       xerr=data[f"{index_type}_conf"],
                       fmt='o', color=colors[output],
                       ecolor='gray', capsize=5)
            
            # Formatting
            ax.set_xlim(left=0)
            ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
            
            # Only add y-labels to leftmost plots
            if j == 0:
                ax.set_ylabel(output, fontsize=12)
            else:
                ax.set_yticklabels([])
                
            # Add titles to top row
            if i == 0:
                ax.set_title(titles[index_type], fontsize=12)
    
    # Add overall title
    fig.suptitle('Sobol Sensitivity Analysis Results', fontsize=14, y=1.05)
    
    # Set y-axis labels to parameter names
    if len(outputs) > 1:
        for ax in axes[:,0]:
            ax.set_yticks(range(len(params)))
            ax.set_yticklabels(params)
    else:
        axes[0].set_yticks(range(len(params)))
        axes[0].set_yticklabels(params)
    
    return fig, axes

# Usage:
fig, axes = plot_sobol_results(sobol_all)
plt.tight_layout()
plt.show()

# Close progress bar when done
pbar.close()
# Calculate elapsed time
elapsed_time = time.time() - start_time
print(f"Simulation took {elapsed_time:.2f} seconds, or {elapsed_time/60:.2f} minutes")

