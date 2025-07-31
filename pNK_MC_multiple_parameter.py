import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
from matplotlib.lines import Line2D  # For custom legend
from scipy.stats import skew


# Start timer
start_time = time.time()

# Parameters to vary in grid search
K_values = [6,7]  # Different K values to test
chi_markup_values = [0.05, 0.075, 0.1,0.125,0.175,0.2,0.225]  # Different chi_markup values to test
n_simulations = 20
T_max = 600

# Other fixed parameters
N = 15 # Number of components (NK landscape)
alpha = 1 # Intensity of the interactions (NK landscape)
h = 6 # Correlation degree (NK landscape) (h=N : -1 correl; h=0 : 1 correl)
J = 20 # Number of firms

M = 20 # Firm memory size
initial_markup = 1 # Markup of all firms at t=0
chi_ms = 1 # Market share sensitivity to relative fitness

iota = 1 # R&D probability parameter
iota_long = 0.7 * iota # R&D probability parameter (long-jump)

w = 0.5 # Weight consumers give to sustainability
w_init = w # Initial weight to sustainability (firms)
chi_w = 0.1 # Rate of change in the weight to sustainability preference (firms)

#Model variations
price_rule = 0 # Price rule: 0 == all penetration, 1 == all skiming, 2== mixed
procyclical = 0 # Variable governing markup/price changes: 0 == market share, 1== profit
free_research = False # Is research free?

# SET SEEDS
research_seeds = np.arange(1, n_simulations+1)
seed_landscape = 1
rng_landscape = np.random.RandomState(seed=seed_landscape)

# SET PLOT DETAILS
transparency = 0 # confidence interval transparency

#DETERMINE LANDSCAPE GLOBAL MAXIMA
X1_star = rng_landscape.randint(0, 2, N)
flip_indices = rng_landscape.choice(N, size=h, replace=False)
X2_star = X1_star.copy()
X2_star[flip_indices] = 1 - X2_star[flip_indices] 

# Initialize storage for all parameter combinations
results = {}
score_data = []

# Grid search over parameter combinations
for K_idx, K in enumerate(K_values):
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
    
    for chi_idx, chi_markup in enumerate(chi_markup_values):
        print(f"Running simulation for K={K}, chi_markup={chi_markup}")
        
        # Set the current parameter combination
        chi_p = chi_markup * initial_markup / (1 + initial_markup)
        
        # Initialize storage for this parameter combination
        all_weighted_price = np.zeros((n_simulations, T_max))
        all_weighted_markup = np.zeros((n_simulations, T_max))
        all_weighted_productivity = np.zeros((n_simulations, T_max))
        all_weighted_sustainability = np.zeros((n_simulations, T_max))
        all_weighted_belief = np.zeros((n_simulations, T_max))
        all_weighted_fitness = np.zeros((n_simulations, T_max))
        all_HHI_series = np.zeros((n_simulations, T_max))
        all_units_produced_series = np.zeros((n_simulations, T_max))
        all_total_profit_series = np.zeros((n_simulations, T_max))
        all_weighted_profit_rate_series = np.zeros((n_simulations, T_max))
        all_impact_series = np.zeros((n_simulations, T_max))
        
        all_std_price = np.zeros((n_simulations, T_max))
        all_std_markup = np.zeros((n_simulations, T_max))
        all_std_productivity = np.zeros((n_simulations, T_max))
        all_std_sustainability = np.zeros((n_simulations, T_max))
        all_std_belief = np.zeros((n_simulations, T_max))
        all_std_fitness = np.zeros((n_simulations, T_max))
        all_std_units = np.zeros((n_simulations, T_max))
        all_std_profit = np.zeros((n_simulations, T_max))
        all_std_profit_rate = np.zeros((n_simulations, T_max))
        all_std_impact = np.zeros((n_simulations, T_max))
        all_std_market_share = np.zeros((n_simulations, T_max))

        all_skew_price = np.zeros((n_simulations, T_max))
        all_skew_markup = np.zeros((n_simulations, T_max))
        all_skew_productivity = np.zeros((n_simulations, T_max))
        all_skew_sustainability = np.zeros((n_simulations, T_max))
        all_skew_belief = np.zeros((n_simulations, T_max))
        all_skew_fitness = np.zeros((n_simulations, T_max))
        all_skew_units = np.zeros((n_simulations, T_max))
        all_skew_profit = np.zeros((n_simulations, T_max))
        all_skew_profit_rate = np.zeros((n_simulations, T_max))
        all_skew_impact = np.zeros((n_simulations, T_max))
        all_skew_market_share = np.zeros((n_simulations, T_max))

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
            units_produced_series = np.sum(units_history, axis=0)
            total_profit_series = np.sum(profit_history, axis=0)
            impact_series = np.sum(impact_history, axis=0)
            HHI_series = np.array([np.sum(market_share_history[:, t]**2) for t in range(T_max)])

            # Weighted averages (using units produced as weights)
            weights = units_history
            weighted_price = np.sum(price_history * weights, axis=0) / np.sum(weights, axis=0)
            weighted_markup = np.sum(markup_history * weights, axis=0) / np.sum(weights, axis=0)
            weighted_productivity = np.sum(f1_history * weights, axis=0) / np.sum(weights, axis=0)
            weighted_sustainability = np.sum(f2_history * weights, axis=0) / np.sum(weights, axis=0)
            weighted_belief = np.sum(w_history * weights, axis=0) / np.sum(weights, axis=0)
            weighted_fitness = np.sum(fitness_history * weights, axis=0) / np.sum(weights, axis=0)
            weighted_profit_rate = np.sum(profit_rate_history * weights, axis=0) / np.sum(weights, axis=0)

            # Standard deviations
            std_price = np.std(price_history, axis=0)
            std_markup = np.std(markup_history, axis=0)
            std_productivity = np.std(f1_history, axis=0)
            std_sustainability = np.std(f2_history, axis=0)
            std_units = np.std(units_history, axis=0)
            std_profit = np.std(profit_history, axis=0)
            std_profit_rate = np.std(profit_rate_history, axis=0)
            std_fitness = np.std(fitness_history, axis=0)
            std_market_share = np.std(market_share_history, axis=0)
            std_impact = np.std(impact_history, axis = 0)
            std_belief = np.std(w_history, axis = 0)
        
            # Pearson skewness coefficients
            skew_price = skew(price_history, axis=0, bias=False)
            skew_markup = skew(markup_history, axis=0, bias=False)
            skew_productivity = skew(f1_history, axis=0, bias=False)
            skew_sustainability = skew(f2_history, axis=0, bias=False)
            skew_units = skew(units_history, axis=0, bias=False)
            skew_profit = skew(profit_history, axis=0, bias=False)
            skew_profit_rate = skew(profit_rate_history, axis=0, bias=False)
            skew_fitness = skew(fitness_history, axis=0, bias=False)
            skew_market_share = skew(market_share_history, axis=0, bias=False)
            skew_impact = skew(impact_history, axis=0, bias=False)
            skew_belief = skew(w_history, axis=0, bias=False)
        
            # Store results from this simulation
            all_weighted_price[sim,:] = weighted_price
            all_weighted_markup[sim,:] = weighted_markup
            all_weighted_productivity[sim,:] = weighted_productivity
            all_weighted_sustainability[sim,:] = weighted_sustainability
            all_weighted_belief[sim,:] = weighted_belief
            all_weighted_fitness[sim,:] = weighted_fitness
            all_HHI_series[sim,:] = HHI_series
            all_units_produced_series[sim,:] = units_produced_series
            all_total_profit_series[sim,:] = total_profit_series
            all_weighted_profit_rate_series[sim,:] = weighted_profit_rate
            all_impact_series[sim,:] = impact_series
            
            all_std_price[sim,:] = std_price
            all_std_markup[sim,:] = std_markup
            all_std_productivity[sim,:] = std_productivity
            all_std_sustainability[sim,:] = std_sustainability
            all_std_belief[sim,:] = std_belief
            all_std_fitness[sim,:] = std_fitness
            all_std_market_share[sim,:] = std_market_share
            all_std_units[sim,:] = std_units
            all_std_profit[sim,:] = std_profit
            all_std_profit_rate[sim,:] = std_profit_rate
            all_std_impact[sim,:] = std_impact
    
            all_skew_price[sim, :]           = skew_price
            all_skew_markup[sim, :]          = skew_markup
            all_skew_productivity[sim, :]    = skew_productivity
            all_skew_sustainability[sim, :]  = skew_sustainability
            all_skew_belief[sim, :]          = skew_belief
            all_skew_fitness[sim, :]         = skew_fitness
            all_skew_units[sim, :]           = skew_units
            all_skew_profit[sim, :]          = skew_profit
            all_skew_profit_rate[sim, :]     = skew_profit_rate
            all_skew_impact[sim, :]          = skew_impact
            all_skew_market_share[sim, :]    = skew_market_share

        # Calculate statistics across simulations
        def calculate_stats(data):
            median = np.median(data, axis=0)
            lower_ci = np.percentile(data, 5, axis=0)
            upper_ci = np.percentile(data, 95, axis=0)
            return median, lower_ci, upper_ci
        
        # Calculate stats for each variable
        price_median, price_lower, price_upper = calculate_stats(all_weighted_price)
        markup_median, markup_lower, markup_upper = calculate_stats(all_weighted_markup)
        prod_median, prod_lower, prod_upper = calculate_stats(all_weighted_productivity)
        sust_median, sust_lower, sust_upper = calculate_stats(all_weighted_sustainability)
        belief_median, belief_lower, belief_upper = calculate_stats(all_weighted_belief)
        fitness_median, fitness_lower, fitness_upper = calculate_stats(all_weighted_fitness)
        hhi_median, hhi_lower, hhi_upper = calculate_stats(all_HHI_series)
        units_median, units_lower, units_upper = calculate_stats(all_units_produced_series)
        profit_median, profit_lower, profit_upper = calculate_stats(all_total_profit_series)
        profit_rate_median, profit_rate_lower, profit_rate_upper = calculate_stats(all_weighted_profit_rate_series)
        impact_median, impact_lower, impact_upper = calculate_stats(all_impact_series)
        
        std_price_median, std_price_lower, std_price_upper = calculate_stats(all_std_price)
        std_markup_median, std_markup_lower, std_markup_upper = calculate_stats(all_std_markup)
        std_prod_median, std_prod_lower, std_prod_upper = calculate_stats(all_std_productivity)
        std_sust_median, std_sust_lower, std_sust_upper = calculate_stats(all_std_sustainability)
        std_belief_median, std_belief_lower, std_belief_upper = calculate_stats(all_std_belief)
        std_fitness_median, std_fitness_lower, std_fitness_upper = calculate_stats(all_std_fitness)
        std_market_share_median, std_market_share_lower, std_market_share_upper = calculate_stats(all_std_market_share)
        std_units_median, std_units_lower, std_units_upper = calculate_stats(all_std_units)
        std_profit_median, std_profit_lower, std_profit_upper = calculate_stats(all_std_profit)
        std_profit_rate_median, std_profit_rate_lower, std_profit_rate_upper = calculate_stats(all_std_profit_rate)
        std_impact_median, std_impact_lower, std_impact_upper = calculate_stats(all_std_impact)
        
        skew_price_median, skew_price_lower, skew_price_upper = calculate_stats(all_skew_price)
        skew_markup_median, skew_markup_lower, skew_markup_upper = calculate_stats(all_skew_markup)
        skew_prod_median, skew_prod_lower, skew_prod_upper = calculate_stats(all_skew_productivity)
        skew_sust_median, skew_sust_lower, skew_sust_upper = calculate_stats(all_skew_sustainability)
        skew_belief_median, skew_belief_lower, skew_belief_upper = calculate_stats(all_skew_belief)
        skew_fitness_median, skew_fitness_lower, skew_fitness_upper = calculate_stats(all_skew_fitness)
        skew_market_share_median, skew_market_share_lower, skew_market_share_upper = calculate_stats(all_skew_market_share)
        skew_units_median, skew_units_lower, skew_units_upper = calculate_stats(all_skew_units)
        skew_profit_median, skew_profit_lower, skew_profit_upper = calculate_stats(all_skew_profit)
        skew_profit_rate_median, skew_profit_rate_lower, skew_profit_rate_upper = calculate_stats(all_skew_profit_rate)
        skew_impact_median, skew_impact_lower, skew_impact_upper = calculate_stats(all_skew_impact)

        # After all simulations are complete, store results for each parameter combination
        results[(K, chi_markup)] = {
            'price_median': price_median,
            'price_lower': price_lower,
            'price_upper': price_upper,
            'markup_median': markup_median,
            'markup_lower': markup_lower,
            'markup_upper': markup_upper,
            'prod_median': prod_median,
            'prod_lower': prod_lower,
            'prod_upper': prod_upper,
            'sust_median': sust_median,
            'sust_lower': sust_lower,
            'sust_upper': sust_upper,
            'belief_median': belief_median,
            'belief_lower': belief_lower,
            'belief_upper': belief_upper,
            'fitness_median': fitness_median,
            'fitness_lower': fitness_lower,
            'fitness_upper': fitness_upper,
            'hhi_median': hhi_median,
            'hhi_lower': hhi_lower,
            'hhi_upper': hhi_upper,
            'units_median': units_median,
            'units_lower': units_lower,
            'units_upper': units_upper,
            'profit_median': profit_median,
            'profit_lower': profit_lower,
            'profit_upper': profit_upper,
            'profit_rate_median': profit_rate_median,
            'profit_rate_lower': profit_rate_lower,
            'profit_rate_upper': profit_rate_upper,
            'impact_median': impact_median,
            'impact_lower': impact_lower,
            'impact_upper': impact_upper,
        
            # Standard deviation (std)
            'std_price_median': std_price_median,
            'std_price_lower': std_price_lower,
            'std_price_upper': std_price_upper,
            'std_markup_median': std_markup_median,
            'std_markup_lower': std_markup_lower,
            'std_markup_upper': std_markup_upper,
            'std_prod_median': std_prod_median,
            'std_prod_lower': std_prod_lower,
            'std_prod_upper': std_prod_upper,
            'std_sust_median': std_sust_median,
            'std_sust_lower': std_sust_lower,
            'std_sust_upper': std_sust_upper,
            'std_belief_median': std_belief_median,
            'std_belief_lower': std_belief_lower,
            'std_belief_upper': std_belief_upper,
            'std_fitness_median': std_fitness_median,
            'std_fitness_lower': std_fitness_lower,
            'std_fitness_upper': std_fitness_upper,
            'std_market_share_median': std_market_share_median,
            'std_market_share_lower': std_market_share_lower,
            'std_market_share_upper': std_market_share_upper,
            'std_units_median': std_units_median,
            'std_units_lower': std_units_lower,
            'std_units_upper': std_units_upper,
            'std_profit_median': std_profit_median,
            'std_profit_lower': std_profit_lower,
            'std_profit_upper': std_profit_upper,
            'std_profit_rate_median': std_profit_rate_median,
            'std_profit_rate_lower': std_profit_rate_lower,
            'std_profit_rate_upper': std_profit_rate_upper,
            'std_impact_median': std_impact_median,
            'std_impact_lower': std_impact_lower,
            'std_impact_upper': std_impact_upper,
        
            # Skewness
            'skew_price_median': skew_price_median,
            'skew_price_lower': skew_price_lower,
            'skew_price_upper': skew_price_upper,
            'skew_markup_median': skew_markup_median,
            'skew_markup_lower': skew_markup_lower,
            'skew_markup_upper': skew_markup_upper,
            'skew_prod_median': skew_prod_median,
            'skew_prod_lower': skew_prod_lower,
            'skew_prod_upper': skew_prod_upper,
            'skew_sust_median': skew_sust_median,
            'skew_sust_lower': skew_sust_lower,
            'skew_sust_upper': skew_sust_upper,
            'skew_belief_median': skew_belief_median,
            'skew_belief_lower': skew_belief_lower,
            'skew_belief_upper': skew_belief_upper,
            'skew_fitness_median': skew_fitness_median,
            'skew_fitness_lower': skew_fitness_lower,
            'skew_fitness_upper': skew_fitness_upper,
            'skew_market_share_median': skew_market_share_median,
            'skew_market_share_lower': skew_market_share_lower,
            'skew_market_share_upper': skew_market_share_upper,
            'skew_units_median': skew_units_median,
            'skew_units_lower': skew_units_lower,
            'skew_units_upper': skew_units_upper,
            'skew_profit_median': skew_profit_median,
            'skew_profit_lower': skew_profit_lower,
            'skew_profit_upper': skew_profit_upper,
            'skew_profit_rate_median': skew_profit_rate_median,
            'skew_profit_rate_lower': skew_profit_rate_lower,
            'skew_profit_rate_upper': skew_profit_rate_upper,
            'skew_impact_median': skew_impact_median,
            'skew_impact_lower': skew_impact_lower,
            'skew_impact_upper': skew_impact_upper
        }
        
        score = 0
        data = results[(K, chi_markup)]
        
        if data['skew_prod_median'][-1] > 0:
            score += 1/7
        if data['std_prod_median'][-1] >= 0.05:
            score += 1/7
        if data['skew_market_share_median'][-1] > 0:
            score += 1/7
        if 0.15 <= data['hhi_median'][-1] <= 0.35:
            score += 1/7
        if 1.0 < data['markup_median'][-1] < 1.5:
            score += 1/7
        if 0.20 <= data['std_markup_median'][-1] <= 0.40:
            score += 1/7
        if 0.10 <= data['skew_markup_median'][-1] > 0:
            score += 1/7
        
        # Append to score_data with proper keys
        score_data.append({
            'K': K,
            'chi_markup': chi_markup,
            'score': score})


# Plot results for each variable with different colors for each parameter combination
time_axis = np.arange(T_max)

# Define a function to plot Monte Carlo results
def plot_parameter_combinations(time_axis, results_dict, var_key, title, ylabel, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(results_dict)))
    
    for idx, ((K, chi_markup), color) in enumerate(zip(results_dict.keys(), colors)):
        data = results_dict[(K, chi_markup)]
        ax.plot(time_axis, data[var_key], color=color, label=f'K={K}, χ={chi_markup}')
    
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel(ylabel)
    ax.grid(True)
    
    if ax is None:
        plt.tight_layout()
        plt.show()

# Create a list of variables and their data
variables = [
    ('Price', 'price_median', 'price_lower', 'price_upper', 'Weighted Price Dynamics',
     'std_price_median', 'std_price_lower', 'std_price_upper', 'Standard Deviation Price',
     'skew_price_median', 'skew_price_lower', 'skew_price_upper', 'Pearson Skewness Price'),
     
    ('Markup', 'markup_median', 'markup_lower', 'markup_upper', 'Weighted Markup Dynamics',
     'std_markup_median', 'std_markup_lower', 'std_markup_upper', 'Standard Deviation Markup',
     'skew_markup_median', 'skew_markup_lower', 'skew_markup_upper', 'Pearson Skewness Markup'),
     
    ('Productivity', 'prod_median', 'prod_lower', 'prod_upper', 'Weighted Productivity Dynamics',
     'std_prod_median', 'std_prod_lower', 'std_prod_upper', 'Standard Deviation Productivity',
     'skew_prod_median', 'skew_prod_lower', 'skew_prod_upper', 'Pearson Skewness Productivity'),
     
    ('Sustainability', 'sust_median', 'sust_lower', 'sust_upper', 'Weighted Sustainability Dynamics',
     'std_sust_median', 'std_sust_lower', 'std_sust_upper', 'Standard Deviation Sustainability',
     'skew_sust_median', 'skew_sust_lower', 'skew_sust_upper', 'Pearson Skewness Sustainability'),
     
    ('Sustainability Preference', 'belief_median', 'belief_lower', 'belief_upper', 'Weighted Sustainability Preference Dynamics',
     'std_belief_median', 'std_belief_lower', 'std_belief_upper', 'Standard Deviation Sustainability weight',
     'skew_belief_median', 'skew_belief_lower', 'skew_belief_upper', 'Pearson Skewness Sustainability weight'),
     
    ('Fitness', 'fitness_median', 'fitness_lower', 'fitness_upper', 'Weighted Fitness Dynamics',
     'std_fitness_median', 'std_fitness_lower', 'std_fitness_upper', 'Standard Deviation Fitness',
     'skew_fitness_median', 'skew_fitness_lower', 'skew_fitness_upper', 'Pearson Skewness Fitness'),
     
    ('Market Concentration', 'hhi_median', 'hhi_lower', 'hhi_upper', 'Market Concentration (HHI)',
     'std_market_share_median', 'std_market_share_lower', 'std_market_share_upper', 'Standard Deviation Market share',
     'skew_market_share_median', 'skew_market_share_lower', 'skew_market_share_upper', 'Pearson Skewness Market share'),
     
    ('Units Produced', 'units_median', 'units_lower', 'units_upper', 'Total Units Produced',
     'std_units_median', 'std_units_lower', 'std_units_upper', 'Standard Deviation Units produced',
     'skew_units_median', 'skew_units_lower', 'skew_units_upper', 'Pearson Skewness Units produced'),
     
    ('Profits', 'profit_median', 'profit_lower', 'profit_upper', 'Total Industry Profits',
     'std_profit_median', 'std_profit_lower', 'std_profit_upper', 'Standard Deviation Profits',
     'skew_profit_median', 'skew_profit_lower', 'skew_profit_upper', 'Pearson Skewness Profits'),
    
    ('Profit rate', 'profit_rate_median', 'profit_rate_lower', 'profit_rate_upper', 'Weighted Profit Rate Dynamics',
     'std_profit_rate_median', 'std_profit_rate_lower', 'std_profit_rate_upper', 'Standard Deviation Profit Rate',
     'skew_profit_rate_median', 'skew_profit_rate_lower', 'skew_profit_rate_upper', 'Pearson Skewness Profit Rate'),
     
    ('Environmental Impact', 'impact_median', 'impact_lower', 'impact_upper', 'Total Environmental Impact',
     'std_impact_median', 'std_impact_lower', 'std_impact_upper', 'Standard Deviation Impact',
     'skew_impact_median', 'skew_impact_lower', 'skew_impact_upper', 'Pearson Skewness Impact')]

# Create color map for parameter combinations
colors = plt.cm.Set1(np.linspace(0, 1, len(results)))

# Create plots for each variable
for var in variables:
    var_name = var[0]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{var_name} Dynamics', y=1.05, fontsize=14)
    
    # --- Plot median with CI ---
    for idx, (K, chi_markup) in enumerate(results.keys()):
        data = results[(K, chi_markup)]
        ax1.plot(time_axis, data[var[1]], color=colors[idx], label=f'K={K}, χ={chi_markup}')
        ax1.fill_between(time_axis, data[var[2]], data[var[3]], color=colors[idx], alpha=transparency)
    ax1.set_title(var[4])
    ax1.set_xlabel('Time')
    ax1.set_ylabel(var_name)
    ax1.grid(True)
    
    # --- Plot standard deviation with CI ---
    for idx, (K, chi_markup) in enumerate(results.keys()):
        data = results[(K, chi_markup)]
        ax2.plot(time_axis, data[var[5]], color=colors[idx])
        ax2.fill_between(time_axis, data[var[6]], data[var[7]], color=colors[idx], alpha=transparency)
    ax2.set_title(var[8])
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Std. Dev.')
    ax2.grid(True)
    
    # --- Plot skewness with CI ---
    for idx, (K, chi_markup) in enumerate(results.keys()):
        data = results[(K, chi_markup)]
        ax3.plot(time_axis, data[var[9]], color=colors[idx])
        ax3.fill_between(time_axis, data[var[10]], data[var[11]], color=colors[idx], alpha=transparency)
    ax3.set_title(var[12])
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Skewness')
    ax3.grid(True)
    
    # Create a single legend for all subplots
    handles = [Line2D([0], [0], color=colors[i], lw=2) for i in range(len(results))]
    labels = [f'K={K}, χ={chi_markup}' for (K, chi_markup) in results.keys()]
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    
    plt.tight_layout()
    plt.show()

import pandas as pd

# Flatten the variables list to extract all variable names (excluding section titles)
final_variable_names = [
    item for group in variables for item in group[1:]
]

# === Step 1: Extract final values ===
final_values_data = []

for (K, chi_markup), data in results.items():
    row = {'K': K, 'chi_markup': chi_markup}
    for var_name in final_variable_names:
        try:
            row[var_name] = data[var_name][-1]  # Last timestep
        except (KeyError, IndexError):
            row[var_name] = np.nan  # Fill with NaN if missing
    final_values_data.append(row)

# Step 1: Create DataFrame from final values
final_values_df = pd.DataFrame(final_values_data)
final_values_df.set_index(['K', 'chi_markup'], inplace=True)

# Step 2: Create DataFrame from score data
score_df = pd.DataFrame(score_data)
score_df.set_index(['K', 'chi_markup'], inplace=True)

# Step 3: Merge scores into final values
final_values_df['score'] = score_df['score']

# Step 4: Save to CSV
csv_path_values = r"C:\Users\mbassartilore\Documents\Recherche\EPOC\Projects\Mixing innovation policies\Validation_tables.csv"
final_values_df.to_csv(csv_path_values)


import statsmodels.api as sm
# Step 1: Create table with final values at T_max for each (K, chi_markup) and each variable
final_values_data = []

# Extract variable names from the `variables` structure (e.g., 'price_median', etc.)
final_variable_names = [var[1] for var in variables]

for (K, chi_markup), data in results.items():
    row = {'K': K, 'chi_markup': chi_markup}
    for var_name in final_variable_names:
        row[var_name] = data[var_name][-1]  # last value (T_max)
    final_values_data.append(row)

# Step 1: Create DataFrame from score_data
score_df = pd.DataFrame(score_data)
score_df.set_index(['K', 'chi_markup'], inplace=True)

# Step 2: Merge with final_values_df
final_values_df['score'] = score_df['score']

# Create DataFrame
final_values_df = pd.DataFrame(final_values_data)
final_values_df.set_index(['K', 'chi_markup'], inplace=True)

# === Step 2: Run OLS for each variable ===
regression_results = []
for var_name in final_variable_names:
    y = final_values_df[var_name].values
    X = final_values_df.reset_index()[['K', 'chi_markup']]
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    
    # Get coefficients, p-values, and stars
    for param, coef, pval in zip(model.params.index, model.params.values, model.pvalues.values):
        if pval < 0.001:
            star = '***'
        elif pval < 0.01:
            star = '**'
        elif pval < 0.05:
            star = '*'
        elif pval < 0.1:
            star = '.'
        else:
            star = ''
        regression_results.append({
            'variable': var_name,
            'parameter': param,
            'coef': round(coef, 4),
            'p-value': round(pval, 4),
            'significance': star
        })

# Convert to DataFrame
regression_df = pd.DataFrame(regression_results)

# Optional: Pivot for cleaner display (wide format)
regression_wide = regression_df.pivot(index='parameter', columns='variable', values='coef')
regression_wide = regression_wide.round(4)

# Save to CSV: OLS COEFFICIENTS
csv_path_ols = r"C:\Users\mbassartilore\Documents\Recherche\EPOC\Projects\Mixing innovation policies\OLS_coefficients.csv"
regression_df.to_csv(csv_path_ols, index=False)

# Display final output
final_values_df, regression_df

#HEATMAP BASED ON SCORE
import seaborn as sns

df = score_df

plt.figure(figsize=(8, 6))
pivot_table = df.pivot_table(index='K', columns='chi_markup', values='score')
sns.heatmap(pivot_table.sort_index(ascending=True), cmap='viridis', annot=True, cbar_kws={'label': 'Score'})
plt.title('Validation Score across K and chi_markup')
plt.xlabel('chi_markup')
plt.ylabel('K')
plt.tight_layout()
plt.show()


# Calculate elapsed time
elapsed_time = time.time() - start_time
print(f"Simulation took {elapsed_time:.2f} seconds")