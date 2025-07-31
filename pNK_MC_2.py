import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
from scipy.stats import skew

# Start timer
start_time = time.time()

# Parameters
seed_landscape = 1 # Set a seed for landscape seeting
rng_landscape = np.random.RandomState(seed=seed_landscape)
N = 15  # Landscape size
K = 6  # Number of interactions
alpha = 1  # Intensity of the interaction
h = 6  # 6 0 correl # 13 -.25 correl #15 -.5 
J = 20       # Number of firms
M = 20      # Memory size
T_max = 600  # Number of iterations
initial_markup = 1  # Initial markup for all firms
w = 0  # Initial sustainability awareness
chi_ms = 1  # Adjustment speed for replicator dynamics
chi_markup = 0.25   # Markup adjustment sensitivity
iota = 1.4  # R&D efficiency parameter
iota_long = 1  # R&D efficiency parameter for long jumps
chi_w = 0.1  # Adjustment speed for sustainability preference
chi_p = chi_markup * initial_markup / ( 1 + initial_markup ) # Set the adjustmed speed for prices so that it has the same sensitivity to changes in the market share
w_init = w # Initial carbon belief for all firms
price_rule = 0 # 0 = all penetration ; 1 = all skimming ; 2 = mixed
procyclical = 0 # 0 = market share ; 1 = profit 
free_research = False # Use to de-activate finance

# Parameters for Monte Carlo simulation
n_simulations = 4  # Number of Monte Carlo runs
fixed_seed_landscape = 1  # Fixed landscape seed
research_seeds = np.arange(1, n_simulations+1)  # Different research seeds for each run

# Initialize storage for all simulations
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

#CREATE pNK LANDSCAPE
# Define a function to calculate the haming distance from one binary string to another
def hamming_distance(x, y): 
    return np.sum(x != y) 

# Define a function to list the binary strings that are at a given hamming distance 
def neighbors(x, dist=1):
    # Returns all binary vectors at Hamming distance = dist from x
    candidates = [] # Create list of candiadtes
    for indices in itertools.combinations(range(N), dist):
        x_new = x.copy()
        x_new[list(indices)] = 1 - x_new[list(indices)]
        candidates.append(x_new)
    return np.array(candidates)

# --- BUILD INTERACTION MATRIX --- Reciprocal interactions (a_(i,j)=-a(j,i)) in the K-vicinity
def build_interaction_matrix(N, K, alpha):
    A = np.zeros((N, N)) #Create a N X N matrix
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

#Calculate interactions vector
def compute_c_vector(X_star, A):
    return np.array([X_star[i] + np.dot(A[i], X_star) for i in range(N)])

#Calculate fitness contribution of each component
def phi_i(i, X, A, c_vector):
    interaction_sum = np.dot(A[i], X)
    return 1 / (1 + abs(X[i] + interaction_sum - c_vector[i]))

#The thotal fitness is given by the average of the fitness contributions
def fitness(X, A, c_vector):
    return np.mean([phi_i(i, X, A, c_vector) for i in range(N)])

# --- INITIALIZE LANDSCAPES ---
A = build_interaction_matrix(N, K, alpha) # Same interaction matrix in both landscapes

X1_star = rng_landscape.randint(0, 2, N) #Random optimal location for the productivity landscape
flip_indices = rng_landscape.choice(N, size=h, replace=False) #Select subset of components N* that are flipped to define the optimal location of the sustainability landscape
X2_star = X1_star.copy()
X2_star[flip_indices] = 1 - X2_star[flip_indices] 

c1 = compute_c_vector(X1_star, A)
c2 = compute_c_vector(X2_star, A)

# Run Monte Carlo simulations
for sim in (range(n_simulations)):
    # Set seeds - fixed landscape, varying research
    seed_research = research_seeds[sim]
    
    # Set different random seeds for landscape generation and research
    rng_research = np.random.RandomState(seed=seed_research)
    
    # Initialize economic variables
    market_shares = np.ones(J) / J  # Initial equal market shares
    market_share_history = np.zeros((J, T_max)) # Track market shares
    profit_history = np.zeros((J, T_max)) # Track profits
    profit_rate_history = np.zeros((J,T_max)) # Track profit rates
    price_history = np.zeros((J, T_max)) # Track prices
    fitness_history = np.zeros((J, T_max)) # Track fitness
    markup_history = np.zeros((J, T_max)) # Track mark-up
    markup_history[:, 0] = initial_markup  # Initialize markups
    firms = []
    f1_history = np.zeros((J, T_max)) # Track productivity
    f2_history = np.zeros((J, T_max)) # Track sustainability
    w_history = np.zeros((J, T_max))  # Track sustainability preferences
    units_history = np.zeros((J, T_max))  # Track units produced
    impact_history = np.zeros((J, T_max))  # Track environmental impact
    successful_research = np.zeros((J, T_max))  # Track successful search CUMULATIVE
    improved_research = np.zeros((J, T_max))  # Track improved search CUMULATIVE
    attempted_long_jumps = np.zeros((J, T_max))  # Track all attempted long jumps CUMULATIVE
    successful_long_jumps = np.zeros((J, T_max))  # Track successful long jumps CUMULATIVE
    improved_long_jumps = np.zeros((J, T_max))  # Track improved long jumps CUMULATIVE

    for j in range(J):
        x0 = 1 - X1_star  # initial location
        memory = [x0.copy()] # Initialize memory
        timers = [0] # Set technology timer in memory as 0
        firms.append({
            'x': x0,
            'w': w_init,  # Initial sustainability preference
            'memory': memory,
            'timers': timers
        })
        w_history[j, 0] = w  # Initialize weight to sustainability history

    # --- ECONOMIC INTERACTIONS
    for t in range(T_max):
        # First update firm positions based on R&D
        for j in range(J):
            firm = firms[j]
            x_current = firm['x']
            memory = firm['memory']
            
            # Determine if firm gets to search based on previous profits
            if t > 0:
                # Probability to search depends on profits
                search_prob = 1 - np.exp(-iota * np.clip(profit_history[j, t-1], 0, 1))
                gets_to_search = rng_research.rand() < search_prob
            else:
                gets_to_search = True  # All firms search in first period
            
            if gets_to_search or free_research :
                # Step 1-2: Search neighbors
                found = False
                successful_research[j, t] = 1
                for d in [1, 2]:
                    candidates = neighbors(x_current, dist=d)
                    candidates = [x for x in candidates if not any(np.array_equal(x, m) for m in memory)]
                    if candidates:
                        # Adjust search probability for long jumps
                        if d == 2:
                            long_search_prob = 1 - np.exp(iota_long * np.clip(profit_history[j, t-1] if t > 0 else 1, 0, 1))
                            gets_long_jump = rng_research.rand() < long_search_prob
                            attempted_long_jumps[j, t] = 1
                            if gets_long_jump or free_research:
                                x_explore = candidates[rng_research.randint(len(candidates))]
                                successful_long_jumps[j, t] = 1
                                found = True
                                break
                            else:
                                continue  # Skip long jumps if not successful
                        else:
                            x_explore = candidates[rng_research.randint(len(candidates))]
                            found = True
                            break
                if not found:
                    x_explore = x_current.copy()  # No valid moves

                # Step 3: Add to memory
                memory.append(x_explore.copy())
                firm['timers'].append(0)

                # Step 4: Evaluate
                f1 = fitness(x_explore, A, c1)
                f2 = fitness(x_explore, A, c2)
                eval_explore = f1 * (1 + firm['w'] * f2)

                f1_old = fitness(x_current, A, c1)
                f2_old = fitness(x_current, A, c2)
                eval_current = f1_old * (1 + firm['w'] * f2_old)

                # Step 5: Update location if better
                if eval_explore >= eval_current:
                    firm['x'] = x_explore.copy()
                    improved_research[j, t] = 1
                    # Track successful long jumps
                    if found and hamming_distance(x_current, x_explore) == 2:
                        improved_long_jumps[j, t] = 1

            # Steps 6-8: Update timers and memory (happens regardless of search)
            for i in range(len(memory)):
                if np.array_equal(memory[i], firm['x']):
                    firm['timers'][i] = 0
                else:
                    firm['timers'][i] += 1

            # Step 9: Truncate memory if needed
            if len(memory) > M:
            # Get current technology
                current_tech = firm['x']
        
            # Get all neighbors at distance 1
                neighbor_techs = neighbors(current_tech, dist=1)
        
            # Find indices of non-neighbor technologies in memory
                non_neighbor_indices = [
                    i for i in range(len(memory)) 
                    if not any(np.array_equal(memory[i], neighbor) for neighbor in neighbor_techs)
                ]
        
                if non_neighbor_indices:
                    # Among non-neighbors, find the one with the highest timer
                    timers_non_neighbors = [firm['timers'][i] for i in non_neighbor_indices]
                    idx_to_remove = non_neighbor_indices[np.argmax(timers_non_neighbors)]
            
                    memory.pop(idx_to_remove)
                    firm['timers'].pop(idx_to_remove)
                else:
                    # If all technologies in memory are neighbors, fall back to original behavior
                    idx_remove = np.argmax(firm['timers'])
                    memory.pop(idx_remove)
                    firm['timers'].pop(idx_remove)

            f1_history[j, t] = fitness(firm['x'], A, c1)
            f2_history[j, t] = fitness(firm['x'], A, c2)
        
        # Then calculate economic variables
        f1_values = np.array([fitness(firms[j]['x'], A, c1) for j in range(J)])
        f2_values = np.array([fitness(firms[j]['x'], A, c2) for j in range(J)])
        
        # Calculate percentage change in market shares for markup adjustment
        if t > 1:
            if procyclical == 0:
                changes = (market_share_history[:, t-1] - market_share_history[:, t-2]) / (market_share_history[:, t-2] + 1e-10)
            else:
                changes = (profit_history[:, t-1] - profit_history[:, t-2]) / (profit_history[:, t-2] + 1e-10)
        else:
            changes = np.zeros(J)
        
        # Calculate market shares (assuming market_shares is available)
        current_market_shares = market_share_history[:, t-1] if t > 0 else np.ones(J)/J
        
        # Update prices and markups differently based on market share
        for j in range(J):
            if t > 0:
                if price_rule == 1 or (current_market_shares[j] > (1/J) and price_rule == 2):  # LEADER
                    # Leader: adjust price directly, then derive markup
                    new_price = price_history[j, t-1] * (1 + chi_p * changes[j])
                    new_price = np.clip(new_price, 0, 50)  # Apply price bounds
                    price_history[j, t] = new_price
                    markup_history[j, t] = price_history[j, t] * f1_values[j] - 1
                    #print(f"Firm {j} is a leader with {current_market_shares[j]}% market share")
                else:  # INCUMBENT
                    # Incumbent: adjust markup directly, then calculate price
                    new_markup = markup_history[j, t-1] * (1 + chi_markup * changes[j])
                    new_markup = np.clip(new_markup, 0, 50)  # Apply markup bounds
                    markup_history[j, t] = new_markup
                    price_history[j, t] = (1 + markup_history[j, t]) / f1_values[j]
                    #print(f"Firm {j} is an incumbent with {current_market_shares[j]}% market share")
            else:
                # Initial period - same for all firms
                markup_history[j, t] = initial_markup
                price_history[j, t] = (1 + initial_markup) / f1_values[j]
        
        # Current prices and markups
        current_prices = price_history[:, t]
        current_markups = markup_history[:, t]
    
        # Calculate prices
        prices = (1 + current_markups) / f1_values
        
        # Calculate firm fitness
        firm_fitness = f1_values * (1 + w * f2_values) / (1 + current_markups)
        
        # Calculate average fitness (using previous market shares)
        if t > 0:
            avg_fitness = np.sum(firm_fitness * market_share_history[:, t-1])
        else:
            avg_fitness = np.mean(firm_fitness)  # First period
        
        # Update market shares (replicator dynamics)
        if t > 0:
            new_market_shares = market_share_history[:, t-1] * (1 + chi_ms * (firm_fitness / avg_fitness - 1))
        else:
            new_market_shares = market_shares * (1 + chi_ms * (firm_fitness / avg_fitness - 1))
        
        new_market_shares = np.clip(new_market_shares, 0.001, 0.999)  # Prevent extinction
        new_market_shares = new_market_shares / np.sum(new_market_shares)  # Normalize
        
        # Calculate variables for statistics
        units_produced = new_market_shares / prices
        impact = units_produced * (1 - f2_values)
        profits = units_produced * (current_markups / f1_values)
        profit_rate = profits / units_produced
        
        # Store results
        market_share_history[:, t] = new_market_shares
        profit_history[:, t] = profits
        profit_rate_history[:, t] = profit_rate
        price_history[:, t] = prices
        fitness_history[:, t] = firm_fitness
        units_history[:, t] = units_produced
        impact_history[:, t] = impact
        
        # Update firms' sustainability preferences (w_j)
        if t > 0:  # Only start adjusting after first period
            current_price = price_history[:, t]
            current_sustainability = f2_history[:, t]
            current_market_shares = market_share_history[:, t]
            w_j = firms[j]['w'] 
            
            for j in range(J):
                # Condition a: Find competitors with higher market share
                competitors = [h for h in range(J) 
                             if (current_market_shares[h] > current_market_shares[j])]
                
                # Filter competitors based on conditions b and c
                valid_competitors = []
                for h in competitors:
                    # Condition b: h does not dominate j
                    cond_b = not ((current_price[h] < current_price[j]) and 
                                 (current_sustainability[h] > current_sustainability[j]))
                    # Condition c: j does not dominate h
                    cond_c = not ((current_price[h] > current_price[j]) and 
                                 (current_sustainability[h] < current_sustainability[j]))
                    # Condition d: Competitor has lower fitness with j's weight to sustainability
                    cond_d = ( 1 + w_j * current_sustainability[j]) / current_price[j] > ( 1 + w_j * current_sustainability[h]) / current_price[h]
                    
                    if cond_b and cond_c and cond_d:
                        valid_competitors.append(h)
                
                # If valid competitors exist, find the one with closest price
                if valid_competitors:
                    # Find competitor with minimal price difference
                    h_star = valid_competitors[np.argmin(np.abs(current_price[valid_competitors] - current_price[j]))]
                    
                    # Update sustainability preference
                    w_j = firms[j]['w']
                    w_h = firms[h_star]['w']
                    sustainability_ratio = current_sustainability[h_star] / (current_sustainability[j] + 1e-10)
                    new_w = w_j * (1 + chi_w * (sustainability_ratio - 1))
                    #new_w = np.clip(new_w, 0, 1)  # Keep w in [0,1] range
                    firms[j]['w'] = new_w
                
                # Record current w value
                w_history[j, t] = firms[j]['w']
        else:
            # In first period, just record initial w values
            for j in range(J):
                w_history[j, t] = firms[j]['w']

    # Calculate time series (totals)
    units_produced_series = np.sum(units_history, axis=0)
    total_profit_series = np.sum(profit_history, axis=0)
    impact_series = np.sum(impact_history, axis = 0)

    # Market concentration measures
    HHI_series = np.array([np.sum(market_share_history[:, t]**2)  for t in range(T_max)])

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

# Generate all triple plots
time_axis = np.arange(T_max)

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

def plot_monte_carlo_results(time_axis, median, lower, upper, title, ylabel, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(time_axis, median, label='Median', color='blue')
    ax.fill_between(time_axis, lower, upper, alpha=0.3, color='blue', label='95% CI')
    ax.set_title(title)
    ax.set_xlabel('Time')
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True)
    
    if ax is None:
        plt.tight_layout()
        plt.show()

# Create a list of variables and their titles
variables = [
    ('Price', price_median, price_lower, price_upper, 'Weighted Price Dynamics', 
     std_price_median, std_price_lower, std_price_upper, 'Standard Deviation Price',
     skew_price_median, skew_price_lower, skew_price_upper, 'Pearson Skewness Price'),
     
    ('Markup', markup_median, markup_lower, markup_upper, 'Weighted Markup Dynamics',
     std_markup_median, std_markup_lower, std_markup_upper, 'Standard Deviation Markup',
     skew_markup_median, skew_markup_lower, skew_markup_upper, 'Pearson Skewness Markup'),
     
    ('Productivity', prod_median, prod_lower, prod_upper, 'Weighted Productivity Dynamics',
     std_prod_median, std_prod_lower, std_prod_upper, 'Standard Deviation Productivity',
     skew_prod_median, skew_prod_lower, skew_prod_upper, 'Pearson Skewness Productivity'),
     
    ('Sustainability', sust_median, sust_lower, sust_upper, 'Weighted Sustainability Dynamics',
     std_sust_median, std_sust_lower, std_sust_upper, 'Standard Deviation Sustainability',
     skew_sust_median, skew_sust_lower, skew_sust_upper, 'Pearson Skewness Sustainability'),
     
    ('Sustainability Preference', belief_median, belief_lower, belief_upper, 'Weighted Sustainability Preference Dynamics',
     std_belief_median, std_belief_lower, std_belief_upper, 'Standard Deviation Sustainability weight',
     skew_belief_median, skew_belief_lower, skew_belief_upper, 'Pearson Skewness Sustainability weight'),
     
    ('Fitness', fitness_median, fitness_lower, fitness_upper, 'Weighted Fitness Dynamics',
     std_fitness_median, std_fitness_lower, std_fitness_upper, 'Standard Deviation Fitness',
     skew_fitness_median, skew_fitness_lower, skew_fitness_upper, 'Pearson Skewness Fitness'),
     
    ('Market Concentration', hhi_median, hhi_lower, hhi_upper, 'Market Concentration (HHI)',
     std_market_share_median, std_market_share_lower, std_market_share_upper, 'Standard Deviation Market share',
     skew_market_share_median, skew_market_share_lower, skew_market_share_upper, 'Pearson Skewness Market share'),
     
    ('Units Produced', units_median, units_lower, units_upper, 'Total Units Produced',
     std_units_median, std_units_lower, std_units_upper, 'Standard Deviation Units produced',
     skew_units_median, skew_units_lower, skew_units_upper, 'Pearson Skewness Units produced'),
     
    ('Profits', profit_median, profit_lower, profit_upper, 'Total Industry Profits',
     std_profit_median, std_profit_lower, std_profit_upper, 'Standard Deviation Profits',
     skew_profit_median, skew_profit_lower, skew_profit_upper, 'Pearson Skewness Profits'),
    
    ('Profit rate', profit_rate_median, profit_rate_lower, profit_rate_upper, 'Weighted Profit Rate Dynamics',
     std_profit_rate_median, std_profit_rate_lower, std_profit_rate_upper, 'Standard Deviation Profit Rate',
     skew_profit_rate_median, skew_profit_rate_lower, skew_profit_rate_upper, 'Pearson Skewness Profit Rate'),
     
    ('Environmental Impact', impact_median, impact_lower, impact_upper, 'Total Environmental Impact',
     std_impact_median, std_impact_lower, std_impact_upper, 'Standard Deviation Impact',
     skew_impact_median, skew_impact_lower, skew_impact_upper, 'Pearson Skewness Impact')
]

# Plot each variable in a 1x3 grid
for var in variables:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{var[0]} Dynamics', y=1.05, fontsize=14)
    
    # Plot weighted/total value
    plot_monte_carlo_results(time_axis, var[1], var[2], var[3], var[4], var[0], ax=ax1)
    
    # Plot standard deviation
    plot_monte_carlo_results(time_axis, var[5], var[6], var[7], var[8], f'Std of {var[0]}', ax=ax2)
    
    # Plot skewness
    plot_monte_carlo_results(time_axis, var[9], var[10], var[11], var[12], f'Skew of {var[0]}', ax=ax3)
    
    plt.tight_layout()
    plt.show()

# Calculate elapsed time
elapsed_time = time.time() - start_time
print(f"Simulation took {elapsed_time:.2f} seconds")