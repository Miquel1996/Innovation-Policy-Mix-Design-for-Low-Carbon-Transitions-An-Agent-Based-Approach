import numpy as np
import matplotlib.pyplot as plt
import itertools
import pandas as pd
from scipy.stats import skew
import time
from tqdm import tqdm  
import multiprocessing
from multiprocessing import Pool
import sys


class ABMController:
    def __init__(self, params, seed_landscape=None, seed_research=None):
        self.params = params
        self.rng_landscape = np.random.RandomState(seed_landscape if seed_landscape is not None else 1)
        self.rng_research = np.random.RandomState(seed_research if seed_research is not None else 1)

    def run(self):
        start_time = time.time()
        results = self._simulate(self.params)
        elapsed = time.time() - start_time
        print(f"✅ Simulation completed in {elapsed:.2f} seconds ({elapsed/60:.2f} min).")
        return results

    def _simulate(self, p):
        # Unpack parameters
        T_max = p['T_max']
        N, K, alpha, h = p['N'], p['K'], p['alpha'], p['h']
        J, M, iota, iota_long_ratio = p['J'], p['M'], p['iota'], p['iota_long_ratio']
        chi_ms, initial_markup, chi_markup = p['chi_ms'], p['initial_markup'], p['chi_markup']
        w_final, w_init, Tstart_w, Tend_w, chi_w = p['w_final'], p['w_init'], p['Tstart_w'], p['Tend_w'], p['chi_w']
        price_rule, procyclical, free_research = p['price_rule'], p['procyclical'], p['free_research']

        chi_p = chi_markup * initial_markup / (1 + initial_markup)
        w_increment = (w_final - w_init) if Tstart_w == Tend_w else (w_final - w_init) / (Tend_w - Tstart_w)
        iota_long = iota * iota_long_ratio

        # --- Landscape functions ---
        def hamming_distance(x, y): return np.sum(x != y)

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
                upper_thresh = i + K // 2 + (0.5 if i % 2 == 1 else 0)
                lower_thresh = i - K // 2 + (-0.5 if i % 2 == 0 else 0)
                for j in range(N):
                    if j == i:
                        A[i, j] = 1
                    elif j > i:
                        A[i, j] = alpha if j <= upper_thresh else (-alpha if j >= N + lower_thresh else 0)
                    else:
                        A[i, j] = -alpha if j >= lower_thresh else (alpha if j <= upper_thresh - N else 0)
            return A

        def compute_c_vector(X_star, A):
            return np.array([X_star[i] + np.dot(A[i], X_star) for i in range(N)])

        def phi_i(i, X, A, c_vector):
            return 1 / (1 + abs(X[i] + np.dot(A[i], X) - c_vector[i]))

        def fitness(X, A, c_vector):
            return np.mean([phi_i(i, X, A, c_vector) for i in range(N)])

        # --- Initialize landscapes ---
        A = build_interaction_matrix(N, K, alpha)
        X1_star = self.rng_landscape.randint(0, 2, N)
        flip_indices = self.rng_landscape.choice(N, size=h, replace=False)
        X2_star = X1_star.copy()
        X2_star[flip_indices] = 1 - X2_star[flip_indices]
        c1, c2 = compute_c_vector(X1_star, A), compute_c_vector(X2_star, A)

        # --- Initialize firms and histories ---
        firms = [{'x': 1 - X1_star, 'w': w_init, 'memory': [1 - X1_star], 'timers': [0]} for _ in range(J)]
        histories = self._initialize_histories(J, T_max, initial_markup, w_init)

        # --- Simulation loop with progress bar ---
        for t in tqdm(range(T_max), desc="Simulating", ncols=80):
            self._time_step(t, firms, histories, A, c1, c2, p, w_increment, fitness, neighbors, hamming_distance)

        return histories

    def _initialize_histories(self, J, T_max, initial_markup, w_init):
        return {
            "market_share": np.zeros((J, T_max), dtype=float),
            "sq_market_share": np.zeros((J, T_max), dtype=float),
            "profit": np.zeros((J, T_max), dtype=float),
            "profit_rate": np.zeros((J, T_max), dtype=float),
            "fitness": np.zeros((J, T_max), dtype=float),
            "price": np.zeros((J, T_max), dtype=float),
            "markup": np.full((J, T_max), float(initial_markup), dtype=float),
            "productivity": np.zeros((J, T_max), dtype=float),
            "sustainability": np.zeros((J, T_max), dtype=float),
            "belief": np.full((J, T_max), float(w_init), dtype=float),
            "units": np.zeros((J, T_max), dtype=float),
            "impact": np.zeros((J, T_max), dtype=float),
            "successful_research": np.zeros((J, T_max), dtype=float),
            "improved_research": np.zeros((J, T_max), dtype=float),
            "attempted_long_jumps": np.zeros((J, T_max), dtype=float),
            "successful_long_jumps": np.zeros((J, T_max), dtype=float),
            "improved_long_jumps": np.zeros((J, T_max), dtype=float)
        }


    # (Keep your full `_time_step` implementation unchanged here)


    def _time_step(self, t, firms, histories, A, c1, c2, params, w_increment, fitness, neighbors, hamming_distance):
        """
        Executes one simulation step: 
        - Firm R&D and innovation
        - Price/markup adjustments
        - Market share updates (replicator dynamics)
        - Belief (sustainability preference) imitation
        - Tracking of histories (R&D success, productivity, etc.)
        """
    
        # === 1. UNPACK PARAMETERS ===
        J = params['J']                  # Number of firms
        M = params['M']                  # Memory size
        chi_ms = params['chi_ms']        # Speed of replicator dynamics
        chi_markup = params['chi_markup'] # Speed of markup adjustment
        initial_markup = params['initial_markup'] # Initial markup for all firms
        chi_w = params['chi_w']          # Speed of belief imitation
        price_rule = params['price_rule']
        procyclical = params['procyclical']
        free_research = params['free_research']
        w_final = params['w_final']
        w_init = params['w_init']
        Tstart_w = params['Tstart_w']
        iota = params['iota']            # R&D efficiency
        iota_long_ratio = params['iota_long_ratio']
    
        chi_p = chi_markup * initial_markup / (1 + initial_markup)
        iota_long = iota * iota_long_ratio
        rng_research = self.rng_research
    
        # === 2. HISTORY REFERENCES ===
        market_share_hist = histories['market_share']
        sq_market_share_hist = histories['sq_market_share']
        fitness_hist = histories['fitness']
        profit_hist = histories['profit']
        profit_rate_hist = histories['profit_rate']
        price_hist = histories['price']
        markup_hist = histories['markup']
        prod_hist = histories['productivity']
        sust_hist = histories['sustainability']
        belief_hist = histories['belief']
        units_hist = histories['units']
        impact_hist = histories['impact']
    
        # === 3. UPDATE GLOBAL SUSTAINABILITY PREFERENCE (w) ===
        w = max(min(w_final, w_init + w_increment * max(0, t - Tstart_w)), 0)
        # -------------------------------------------------------------------------
        # === 4. INNOVATION & R&D SEARCH (Bounded Rationality) ===
        # Each firm may attempt local or long jumps in the NK landscape
        # -------------------------------------------------------------------------
        for j in range(J):
            firm = firms[j]
            x_current = firm['x']
            memory = firm['memory']
    
            # ---- R&D Participation Decision ----
            if t > 0:
                # Probability to engage in R&D depends on previous profit
                search_prob = 1 - np.exp(-iota * np.clip(profit_hist[j, t - 1], 0, 1))
                gets_to_search = rng_research.rand() < search_prob
            else:
                gets_to_search = True  # First period: everyone searches
    
            if gets_to_search or free_research:
                histories["successful_research"][j, t] = 1  # Track research attempt
                found = False
    
                # ---- Stepwise Search: Local (d=1) then Long Jump (d=2) ----
                for d in [1, 2]:
                    candidates = neighbors(x_current, dist=d)
                    candidates = [x for x in candidates if not any(np.array_equal(x, m) for m in memory)]
    
                    if candidates:
                        if d == 2:  # Long jumps
                            histories["attempted_long_jumps"][j, t] = 1
                            long_search_prob = 1 - np.exp(-iota_long * np.clip(profit_hist[j, t-1] if t > 0 else 1, 0, 1))
                            if rng_research.rand() < long_search_prob or free_research:
                                x_explore = candidates[rng_research.randint(len(candidates))]
                                histories["successful_long_jumps"][j, t] = 1
                                found = True
                                break
                            else:
                                continue
                        else:  # Local search
                            x_explore = candidates[rng_research.randint(len(candidates))]
                            found = True
                            break
    
                # If no candidate found, remain at current position
                if not found:
                    x_explore = x_current.copy()
    
                # ---- Add to Memory and Evaluate Fitness ----
                memory.append(x_explore.copy())
                firm['timers'].append(0)
    
                f1_new, f2_new = fitness(x_explore, A, c1), fitness(x_explore, A, c2)
                eval_new = f1_new ** (1 - firm['w']) * f2_new ** firm['w']
    
                f1_old, f2_old = fitness(x_current, A, c1), fitness(x_current, A, c2)
                eval_old = f1_old ** (1 - firm['w']) * f2_old ** firm['w']
    
                # ---- Adopt New Tech if Better ----
                if eval_new >= eval_old:
                    firm['x'] = x_explore.copy()
                    histories["improved_research"][j, t] = 1
                    if hamming_distance(x_current, x_explore) == 2:
                        histories["improved_long_jumps"][j, t] = 1
    
            # ---- Memory Management ----
            for i in range(len(memory)):
                firm['timers'][i] = 0 if np.array_equal(memory[i], firm['x']) else firm['timers'][i] + 1
    
            if len(memory) > M:
                current_tech = firm['x']
                neighbor_techs = neighbors(current_tech, dist=1)
                non_neighbors = [i for i in range(len(memory)) if not any(np.array_equal(memory[i], n) for n in neighbor_techs)]
                idx_to_remove = (non_neighbors[np.argmax([firm['timers'][i] for i in non_neighbors])]
                                 if non_neighbors else np.argmax(firm['timers']))
                memory.pop(idx_to_remove)
                firm['timers'].pop(idx_to_remove)
    
            # ---- Record Fitness Histories ----
            prod_hist[j, t] = fitness(firm['x'], A, c1)
            sust_hist[j, t] = fitness(firm['x'], A, c2)
    
        # -------------------------------------------------------------------------
        # === 5. ECONOMIC DYNAMICS ===
        # Firms adjust prices/markups based on market share or profit signals
        # -------------------------------------------------------------------------
        f1_vals = np.array([fitness(f['x'], A, c1) for f in firms])
        f2_vals = np.array([fitness(f['x'], A, c2) for f in firms])
    
        # Market share or profit-driven changes
        if t > 1:
            if procyclical == 0:
                changes = (market_share_hist[:, t-1] - market_share_hist[:, t-2]) / (market_share_hist[:, t-2] + 1e-10)
            else:
                changes = (profit_hist[:, t-1] - profit_hist[:, t-2]) / (profit_hist[:, t-2] + 1e-10)
        else:
            changes = np.zeros(J)
    
        current_market_shares = market_share_hist[:, t-1] if t > 0 else np.ones(J) / J
    
        # ---- Price and Markup Adjustments ----
        for j in range(J):
            if t > 0:
                if price_rule == 1 or (current_market_shares[j] > (1/J) and price_rule == 2):  # Leader
                    new_price = price_hist[j, t-1] * (1 + chi_p * changes[j])
                    new_price = np.clip(new_price, 0, 50)
                    price_hist[j, t] = new_price
                    markup_hist[j, t] = price_hist[j, t] * f1_vals[j] - 1
                else:  # Incumbent
                    new_markup = markup_hist[j, t-1] * (1 + chi_markup * changes[j])
                    markup_hist[j, t] = new_markup
                    price_hist[j, t] = (1 + markup_hist[j, t]) / f1_vals[j]
            else:
                markup_hist[j, t] = initial_markup
                price_hist[j, t] = (1 + initial_markup) / f1_vals[j]
        
        # ✅ Explicitly refresh current prices/markups
        current_prices = price_hist[:, t]
        current_markups = markup_hist[:, t]
        
        # ---- Fitness, Replicator Dynamics ----
        firm_fitness = f2_vals ** w / current_prices ** (1 - w)
        avg_fitness = np.sum(firm_fitness * current_market_shares) if t > 0 else np.mean(firm_fitness)
        
        new_market_shares = (current_market_shares * (1 + chi_ms * (firm_fitness / avg_fitness - 1)))
        new_market_shares = np.clip(new_market_shares, 0.001, 0.999)
        new_market_shares /= np.sum(new_market_shares)
        
        # Units, impact, profits using refreshed markups
        units = new_market_shares / current_prices
        impact = units * (1 - f2_vals)
        profits = units * (current_markups / f1_vals)
        profit_rate = profits / units # or markup / TFP
        sq_market_share = new_market_shares ** 2
        
        # Save to histories
        market_share_hist[:, t] = new_market_shares
        sq_market_share_hist[:, t] = sq_market_share
        price_hist[:, t] = current_prices
        profit_hist[:, t] = profits
        units_hist[:, t] = units
        impact_hist[:, t] = impact
        fitness_hist[:,t] = firm_fitness
        profit_rate_hist[:, t] = profit_rate 

        # -------------------------------------------------------------------------
        # === 7. BELIEF IMITATION (Sustainability Preference Dynamics) ===
        # Firms observe higher-share rivals and imitate sustainability weights (w)
        # -------------------------------------------------------------------------
        for j in range(J):
            w_j = firms[j]['w']
            competitors = [h for h in range(J) if new_market_shares[h] > new_market_shares[j]]
            valid_competitors = []
    
            for h in competitors:
                cond_b = not ((current_prices[h] < current_prices[j]) and (sust_hist[h, t] > sust_hist[j, t]))
                cond_c = not ((current_prices[h] > current_prices[j]) and (sust_hist[h, t] < sust_hist[j, t]))
                cond_d = (sust_hist[j, t] ** w_j / current_prices[j] ** (1 - w_j)) > (sust_hist[h, t] ** w_j / current_prices[h] ** (1 - w_j))
                if cond_b and cond_c and cond_d:
                    valid_competitors.append(h)
    
            if valid_competitors:
                h_star = valid_competitors[np.argmin(np.abs(current_prices[valid_competitors] - current_prices[j]))]
                new_w = w_j * (1 + chi_w * (sust_hist[h_star, t] / (sust_hist[j, t] + 1e-10) - 1))
                firms[j]['w'] = new_w
    
            belief_hist[j, t] = firms[j]['w']

class Statistician:
    def __init__(self, results, params):
        """
        Auto-detects whether results correspond to a single run (dict) or multiple runs (list of dicts).
        """
        self.params = params
        if isinstance(results, list):
            self.results_list = results
            self.multi_run = True
        else:
            self.results_list = [results]
            self.multi_run = False

    def compute(self, ci=95):
        """
        Automatically computes either single-run metrics or Monte Carlo (multi-run) metrics.
        """
        if self.multi_run:
            self.metrics = self.compute_mc_metrics(ci=ci)
        else:
            self.metrics = self.compute_metrics(self.results_list[0])
        return self.metrics

    def plot(self):
        """
        Automatically plots either single-run metrics or Monte Carlo metrics.
        """
        if self.multi_run:
            self.plot_mc_metrics(self.metrics)
        else:
            self.plot_metrics(self.metrics)

    # === Existing Single-Run Metrics ===
    def compute_metrics(self, histories):
        weights = histories['units']
        T_max = weights.shape[1]

        metrics = {}
        def weighted_avg(var): return np.sum(histories[var] * weights, axis=0) / np.sum(weights, axis=0)
        def avg(var): return np.mean(histories[var], axis=0)
        def std_from_hist(var): return np.std(histories[var], axis=0)
        def skew_from_hist(var):
            skews = []
            for t in range(T_max):
                vals = histories[var][:, t]
                if np.allclose(vals, vals[0]):  # all identical
                    skews.append(0.0)
                else:
                    skews.append(skew(vals, bias=False))
            return np.array(skews)

        metrics['market_share'] = {'avg': avg('market_share'),
                                   'mv': np.array([np.sum(histories['market_share'][:, t]**2) for t in range(T_max)])}
        metrics['profit'] = {'avg': avg('profit'), 'mv': weighted_avg('profit')}
        metrics['impact'] = {'avg': avg('impact'), 'mv': weighted_avg('impact')}
        metrics['productivity'] = {'avg': avg('productivity'), 'mv': weighted_avg('productivity')}
        metrics['sustainability'] = {'avg': avg('sustainability'), 'mv': weighted_avg('sustainability')}
        metrics['markup'] = {'avg': avg('markup'), 'mv': weighted_avg('markup')}
        metrics['price'] = {'avg': avg('price'), 'mv': weighted_avg('price')}
        metrics['belief'] = {'avg': avg('belief'), 'mv': weighted_avg('belief')}
        metrics['units'] = {'avg': avg('units'), 'mv': weighted_avg('units')}
        metrics['profit_rate'] = {'avg': avg('profit_rate'), 'mv': weighted_avg('profit_rate')}
        metrics['fitness'] = {'avg': avg('fitness'), 'mv': weighted_avg('fitness')}

        for var in ['market_share','profit','impact','productivity','sustainability','markup','price','belief','units','profit_rate','fitness']:
            metrics[var]['std'] = std_from_hist(var)
            metrics[var]['skew'] = skew_from_hist(var)

        metrics['R&D_success'] = {
            'avg': self.compute_avg_cumulative_success(histories),
            'final': np.mean(histories['successful_research'][:, -1])
        }

        return metrics

    def compute_avg_cumulative_success(self, histories):
        successful_research = histories['successful_research']
        J, T_max = successful_research.shape
        return np.mean([np.cumsum(successful_research[j]) / np.arange(1, T_max+1) for j in range(J)], axis=0)

    # === Monte Carlo Metrics ===
    def compute_mc_metrics(self, ci=95):
        mc_metrics = {}
        single_metrics_runs = [self.compute_metrics(run) for run in self.results_list]
        var_list = ['market_share','units','profit','profit_rate','impact','productivity','sustainability','markup','price','belief','fitness']

        for var in var_list:
            for key in ['mv','std','skew']:
                all_runs = np.array([m[var][key] for m in single_metrics_runs])
                if key == 'skew':
                    all_runs = np.nan_to_num(all_runs, nan=0.0)  # Clean NaNs
                mc_metrics.setdefault(var, {})[key] = {
                    'median': np.median(all_runs, axis=0),
                    'low': np.percentile(all_runs, (100-ci)/2, axis=0),
                    'high': np.percentile(all_runs, 100-(100-ci)/2, axis=0)
                }

        return mc_metrics

    # === Plot Single Run ===
    def plot_metrics(self, metrics):
        var_list = ['markup', 'productivity', 'market_share', 'sustainability']
        for var in var_list:
            fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharex=True)
            fig.suptitle(f"{var.capitalize()} Dynamics")
            axes[0].plot(metrics[var]['mv'], color='black', label="Main variable")
            axes[1].plot(metrics[var]['std'], color='orange', label="Std Dev")
            axes[2].plot(metrics[var]['skew'], color='green', label="Skewness")
            for ax in axes: ax.grid(True); ax.legend()
            plt.tight_layout()
            plt.show()


    # === Plot Monte Carlo ===
    def plot_mc_metrics(self, mc_metrics):
        for var in mc_metrics.keys():
            fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharex=True)
            fig.suptitle(f"{var.capitalize()} Dynamics (MC Median & CI)")
    
            # Panel 1: Main variable (mv)
            median = mc_metrics[var]['mv']['median']
            low = mc_metrics[var]['mv']['low']
            high = mc_metrics[var]['mv']['high']
            axes[0].plot(median, color='black', label="Median")
            axes[0].fill_between(range(len(median)), low, high, color='gray', alpha=0.3, label="CI")
            axes[0].grid(True)
            axes[0].legend()
    
            # Panel 2: Std Dev
            median = mc_metrics[var]['std']['median']
            low = mc_metrics[var]['std']['low']
            high = mc_metrics[var]['std']['high']
            axes[1].plot(median, color='orange', label="Std Dev (Median)")
            axes[1].fill_between(range(len(median)), low, high, color='gray', alpha=0.3, label="CI")
            axes[1].grid(True)
            axes[1].legend()
    
            # Panel 3: Skewness
            median = mc_metrics[var]['skew']['median']
            low = mc_metrics[var]['skew']['low']
            high = mc_metrics[var]['skew']['high']
            axes[2].plot(median, color='green', label="Skewness (Median)")
            axes[2].fill_between(range(len(median)), low, high, color='gray', alpha=0.3, label="CI")
            axes[2].grid(True)
            axes[2].legend()
            for ax in axes: ax.grid(True); ax.legend()
            plt.tight_layout()
            plt.show()

    # === Summary Table ===
    def summary_table(self, metrics):
        """
        Returns a summary DataFrame with:
        - Final median/mean of each variable.
        - Final std and skew of each variable.
        - Confidence intervals (low/high) for mv, std, skew (MC only).
        - Total productivity growth.
        - Average R&D success.
        Works for both single-run and Monte Carlo metrics.
        """
        rows = []
    
        if self.multi_run:
            # === Monte Carlo Mode ===
            for var in metrics.keys():
                if 'mv' in metrics[var]:
                    final_mv = np.round(metrics[var]['mv']['median'][-1], 4)
                    mv_low = np.round(metrics[var]['mv']['low'][-1], 4)
                    mv_high = np.round(metrics[var]['mv']['high'][-1], 4)
    
                    final_std = np.round(metrics[var]['std']['median'][-1], 4)
                    std_low = np.round(metrics[var]['std']['low'][-1], 4)
                    std_high = np.round(metrics[var]['std']['high'][-1], 4)
    
                    final_skew = np.round(metrics[var]['skew']['median'][-1], 4)
                    skew_low = np.round(metrics[var]['skew']['low'][-1], 4)
                    skew_high = np.round(metrics[var]['skew']['high'][-1], 4)
    
                    rows.append([
                        f"{var}_final", final_mv, final_std, final_skew,
                        mv_low, mv_high,
                        std_low, std_high,
                        skew_low, skew_high
                    ])
    
            # Productivity growth
            prod_median_start = metrics['productivity']['mv']['median'][0]
            prod_median_end = metrics['productivity']['mv']['median'][-1]
            rows.append([
                "productivity_growth",
                np.round((prod_median_end - prod_median_start) / prod_median_start, 4),
                None, None, None, None, None, None, None, None
            ])
            rows.append(["avg_productivity_growth",np.round((prod_median_end - prod_median_start) / prod_median_start / self.params['T_max'],4),
                None, None, None, None, None, None, None, None])
    
            # Average R&D success rate
            successes = [np.mean(run['successful_research'][:, -1]) for run in self.results_list]
            rows.append([
                "avg_R&D_success", np.round(np.mean(successes), 4),
                None, None, None, None, None, None, None, None
            ])
    
        else:
            # === Single Run Mode ===
            for var in metrics.keys():
                if 'mv' in metrics[var]:
                    final_mv = np.round(metrics[var]['mv'][-1], 4)
                    final_std = np.round(metrics[var]['std'][-1], 4)
                    final_skew = np.round(metrics[var]['skew'][-1], 4)
    
                    rows.append([
                        f"{var}_final", final_mv, final_std, final_skew,
                        None, None, None, None, None, None
                    ])
    
            # Productivity growth
            prod_start = metrics['productivity']['mv'][0]
            prod_end = metrics['productivity']['mv'][-1]
            rows.append([
                "productivity_growth",
                np.round((prod_end - prod_start) / prod_start, 4),
                None, None, None, None, None, None, None, None
            ])
            rows.append(["avg_productivity_growth",np.round((prod_end - prod_start) / prod_start / self.params['T_max'],4),
                None, None, None, None, None, None, None, None])
    
            # Average R&D success rate
            rows.append([
                "avg_R&D_success", np.round(metrics['R&D_success']['final'], 4),
                None, None, None, None, None, None, None, None
            ])
    
        # Convert to DataFrame with new columns
        df_summary = pd.DataFrame(
            rows,
            columns=[
                "Metric", "Value", "Std_Final", "Skew_Final",
                "MV_CI_Low", "MV_CI_High",
                "Std_CI_Low", "Std_CI_High",
                "Skew_CI_Low", "Skew_CI_High"
            ]
        )
        return df_summary




class Paralleler:
    @staticmethod
    def run_mc_fixed(params, n_runs, n_processes, overlay_debug=False, auto_plot=True, debug_print=True, variable_landscape=False):
        """
        Runs Monte Carlo simulations.
        - If variable_landscape = False, all runs use the same landscape seed (seed_landscape=1),
          and only the research seed varies.
        - If variable_landscape = True, both seeds vary across runs.
        """
        if variable_landscape:
            seeds = [(i, i) for i in range(1, n_runs + 1)]  # both vary
        else:
            seeds = [(1, i) for i in range(1, n_runs + 1)]  # fixed landscape
    
        print("Seeds for MC runs:", seeds)
        args = [(params, s[0], s[1]) for s in seeds]
        print(f"Starting MC: {n_runs} runs, {n_processes} processes...")
        start_time = time.time()

        if Paralleler._is_interactive():
            print("⚠ Interactive mode detected: running sequentially.")
            results = [Paralleler._run_single(*arg) for arg in tqdm(args)]
        else:
            with Pool(n_processes) as pool:
                results = list(tqdm(pool.starmap(Paralleler._run_single, args), total=n_runs))

        elapsed = time.time() - start_time
        print(f"✅ MC completed in {elapsed:.2f}s ({elapsed/60:.2f} min)")

        if debug_print:
            for run_idx, run in enumerate(results):
                print(f"\n=== RUN {run_idx+1}/{n_runs} ===")
                print("Market share HHI:", [np.sum(run['market_share'][:, t]**2) for t in range(run['market_share'].shape[1])][-5:])
                print("Average markup:", np.mean(run['markup'], axis=0)[-5:])
                print("Total profit:", np.sum(run['profit'], axis=0)[-5:])
                print("Total output:", np.sum(run['units'], axis=0)[-5:])
                print("Average price:", np.mean(run['price'], axis=0)[-5:])
                print("Average productivity:", np.mean(run['productivity'], axis=0)[-5:])

        stats = Statistician(results, params)
        metrics = stats.compute(ci=95)
        if auto_plot:
            stats.plot()

        return results, stats, metrics

    @staticmethod
    def _run_single(params, seed_landscape, seed_research):
        controller = ABMController(params, seed_landscape, seed_research)
        return controller.run()

    @staticmethod
    def _is_interactive():
        import sys
        return hasattr(sys, 'ps1') or 'IPYKERNEL' in sys.modules or (
            hasattr(sys, 'executable') and 'SPYDER' in sys.executable.upper()
        )

debug_params = {
    'T_max': 600, 'N': 15, 'K': 6, 'alpha': 1, 'h': 6,
    'J': 20, 'M': 20, 'iota': 1, 'iota_long_ratio': 0.7,
    'chi_ms': 1, 'initial_markup': 1.0, 'chi_markup': 0.2,
    'w_final': 0.1, 'w_init': 0.01, 'Tstart_w': 240, 'Tend_w': 480, 'chi_w': 0.2,
    'price_rule': 0, 'procyclical': 0, 'free_research': False
}
# === Main execution ===
if __name__ == "__main__":
    # Detect total number of cores and reserve one for system
    available_cores = max(1, multiprocessing.cpu_count() - 1)
    
    results, stats, metrics = Paralleler.run_mc_fixed(
        debug_params,
        n_runs=50,
        n_processes=available_cores,
        overlay_debug=True,
        auto_plot=True,
        debug_print=False
    )

    # ✅ Control display of wide tables
    pd.set_option("display.max_columns", None)    # Show all columns
    pd.set_option("display.width", 200)          # Extend print width
    df_summary = stats.summary_table(metrics)
    print(df_summary)

