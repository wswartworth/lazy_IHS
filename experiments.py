import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
import seaborn as sns
import pandas as pd
import time
import os

# =============================================================================
# 1. Loss Functions and Derivatives (Unified Structure)
# =============================================================================

# --- 1.1 Pseudo-Huber Loss ---

def pseudo_huber_loss(r, sigma=1.0):
    return sigma**2 * (np.sqrt(1 + (r/sigma)**2) - 1)

def pseudo_huber_grad(r, sigma=1.0):
    return r / np.sqrt(1 + (r/sigma)**2)

def pseudo_huber_hess_weights(r, sigma=1.0):
    return (1 + (r/sigma)**2)**(-1.5)

def pseudo_huber_objective(x, A, y, params):
    sigma = params.get('sigma', 1.0)
    residuals = A @ x - y
    loss = pseudo_huber_loss(residuals, sigma)
    return np.mean(loss)

def pseudo_huber_gradient(x, A, y, params):
    sigma = params.get('sigma', 1.0)
    n = A.shape[0]
    residuals = A @ x - y
    psi = pseudo_huber_grad(residuals, sigma)
    return (1/n) * A.T @ psi

def pseudo_huber_hessian_weights_scaled(x, A, y, params):
    sigma = params.get('sigma', 1.0)
    n = A.shape[0]
    residuals = A @ x - y
    H_weights = pseudo_huber_hess_weights(residuals, sigma)
    return np.sqrt(H_weights / n)

# --- 1.2 Logistic Loss ---

def sigmoid(z):
    # Clip z to avoid overflow in exp(-z)
    z = np.clip(z, -50, 50)
    return 1 / (1 + np.exp(-z))

def logistic_objective(x, A, y, params=None):
    h = sigmoid(A @ x)
    # Add a small epsilon to avoid log(0)
    epsilon = 1e-15
    h = np.clip(h, epsilon, 1 - epsilon)
    loss = -(y * np.log(h) + (1 - y) * np.log(1 - h))
    return np.mean(loss)

def logistic_gradient(x, A, y, params=None):
    n = A.shape[0]
    h = sigmoid(A @ x)
    return (1/n) * A.T @ (h - y)

def logistic_hessian_weights_scaled(x, A, y, params=None):
    n = A.shape[0]
    h = sigmoid(A @ x)
    # Logistic regression Hessian weights: h * (1 - h)
    H_weights = h * (1 - h)
    return np.sqrt(H_weights / n)

# --- 1.3 Loss Function Registry ---

LOSS_FUNCTIONS = {
    'pseudo_huber': {
        'objective': pseudo_huber_objective,
        'gradient': pseudo_huber_gradient,
        'hessian_weights_scaled': pseudo_huber_hessian_weights_scaled,
    },
    'logistic': {
        'objective': logistic_objective,
        'gradient': logistic_gradient,
        'hessian_weights_scaled': logistic_hessian_weights_scaled,
    }
}

# =============================================================================
# 2. Data Generation (Unified Structure)
# =============================================================================

def generate_pseudo_huber_data(n, d, params):
    outlier_fraction = params.get('outlier_fraction', 0.1)
    noise_level = params.get('noise_level', 0.5)
    outlier_magnitude = params.get('outlier_magnitude', 20)

    x_true = np.random.randn(d) * 5
    A = np.random.randn(n, d)
    y_clean = A @ x_true + noise_level * np.random.randn(n)
    y = y_clean.copy()
    n_outliers = int(n * outlier_fraction)
    outlier_indices = np.random.choice(n, n_outliers, replace=False)
    y[outlier_indices] += outlier_magnitude * np.sign(np.random.randn(n_outliers))
    return A, y, x_true

def generate_logistic_data(n, d, params):
    outlier_fraction = params.get('outlier_fraction', 0.0)

    x_true = np.random.randn(d) * 5
    A = np.random.randn(n, d)
    z = A @ x_true
    p = sigmoid(z)

    # Generate clean binary labels
    y_clean = (np.random.rand(n) < p).astype(float) # Use float for consistency with loss functions
    y = y_clean.copy()

    # Introduce outliers by flipping labels
    n_outliers = int(n * outlier_fraction)
    if n_outliers > 0:
        outlier_indices = np.random.choice(n, n_outliers, replace=False)
        y[outlier_indices] = 1.0 - y[outlier_indices] # Flip the label

    return A, y, x_true

def generate_year_data(n, d, params):
    global A_year
    global y_year
    if A_year is None or y_year is None:
        raise RuntimeError("YearPredictionMSD data (A_year and y_year) not available.")
    # Return the pre-loaded dataset (ignoring n, d, params as it's a fixed dataset)
    return A_year, y_year, None

DATA_GENERATORS = {
    'pseudo_huber': generate_pseudo_huber_data,
    'logistic': generate_logistic_data,
    'year_prediction_msd' : generate_year_data # Renamed key to be more descriptive
}

# =============================================================================
# 3. Sketching Mechanisms
# =============================================================================

# --- 3.1 SRHT Utilities ---
def next_power_of_2(x):
    # Handle the case where x might be 0
    if x == 0:
        return 1
    return 1 << (x - 1).bit_length()

# --- 3.2 Sketch Implementations ---

def gaussian_sketch(B, m, S_precomputed=None):
    n = B.shape[0]
    S = np.random.randn(m, n) / np.sqrt(m)
    return S @ B

def sparse_random_projection_matrix(n, m):
    # This generates the matrix S explicitly, which might be slow for large n, m
    # but is needed if we want to pre-simulate S for the sparse Lazy-IHS variant.
    S = np.zeros((m, n))
    rows = np.random.randint(0, m, n)
    cols = np.arange(n)
    signs = np.random.choice([-1.0, 1.0], n)
    # Use np.add.at for efficient accumulation when indices repeat
    np.add.at(S, (rows, cols), signs)
    return S

def sparse_sketch(B, m, S_precomputed=None):
    n = B.shape[0]
    if S_precomputed is not None:
        S = S_precomputed
    else:
        # If not precomputed, we compute it on the fly.
        # For true efficiency in standard IHS, specialized matrix-vector products
        # would be better than explicitly forming S, but this unified approach works.
        S = sparse_random_projection_matrix(n, m)
    return S @ B


def sparse_coordinate_sketch(B, m, k, S_precomputed=None):
    """
    Hashes the largest k coordinates into their own buckets,
    and runs a CountSketch on the remaining coordinates.

    Args:
        B (np.ndarray): The matrix to sketch (n x d).
        m (int): The total sketch size (number of rows in the sketch).
                 Must be >= k.
        k (int): The number of top coordinates to isolate.
        S_precomputed (tuple, optional): Precomputed sketch matrices (S_top_k, S_remaining).

    Returns:
        np.ndarray: The sketched matrix (m x d).
    """
    n, d = B.shape

    if m < k:
        raise ValueError(f"Sketch size m ({m}) must be greater than or equal to k ({k}).")

    # 1. Identify the largest k coordinates (rows) based on their L2 norm.
    norms = np.linalg.norm(B, axis=1)
    top_k_indices = np.argpartition(norms, -k)[-k:]
    remaining_indices = np.setdiff1d(np.arange(n), top_k_indices)

    # 2. Initialize the sketched matrix
    SB = np.zeros((m, d))

    # 3. Handle the top k coordinates (identity sketch for the first k rows of SB)
    SB[:k, :] = B[top_k_indices, :]

    # 4. Handle the remaining coordinates (CountSketch for the remaining m-k rows of SB)
    m_remaining = m - k
    if m_remaining > 0:
        B_remaining = B[remaining_indices, :]
        n_remaining = B_remaining.shape[0]

        if S_precomputed is not None:
            S_remaining = S_precomputed
        else:
            # Generate a CountSketch matrix for the remaining coordinates
            S_remaining = sparse_random_projection_matrix(n_remaining, m_remaining)

        SB[k:, :] = S_remaining @ B_remaining

    return SB


def srht_sketch(B, m, S_precomputed=None):
    n, d = B.shape
    n_padded = next_power_of_2(n)

    # Padding if necessary
    if n_padded > n:
        B_pad = np.zeros((n_padded, d))
        B_pad[:n, :] = B
    else:
        # Use B.copy() to avoid modifying the original B if n_padded == n
        B_pad = B.copy()

    # Diagonal matrix D with Rademacher random variables
    D = np.random.choice([-1.0, 1.0], size=n_padded)
    B_flipped = (B_pad.T * D).T

    # Apply Hadamard transform. Note: scipy.linalg.hadamard returns the unnormalized matrix.
    # We need to normalize by 1/sqrt(n_padded) for the standard definition.
    H_B = hadamard(n_padded) @ B_flipped / np.sqrt(n_padded)

    # Subsampling P
    # Ensure m is not larger than n_padded
    m = min(m, n_padded)
    idx = np.random.choice(n_padded, m, replace=False)
    # Rescaling by sqrt(n_padded / m)
    S_B = H_B[idx, :] * np.sqrt(n_padded / m)
    return S_B

SKETCH_FUNCTIONS = {
    'gaussian': gaussian_sketch,
    'sparse': sparse_sketch,
    'srht': srht_sketch,
    'sparse_coordinate': sparse_coordinate_sketch, # Added new sketch type
}

# =============================================================================
# 4. Optimization Algorithms (Unified)
# =============================================================================

# --- 4.1 Armijo Backtracking Line Search (Unified) ---

def armijo_backtracking(x, d, g, A, y, loss_fns, loss_params, alpha=0.3, beta=0.8):
    eta = 1.0
    objective_fn = loss_fns['objective']
    f_x = objective_fn(x, A, y, loss_params)

    # Armijo condition check: f(x + eta*d) <= f(x) + alpha * eta * <g, d>
    armijo_rhs_base = alpha * np.dot(g, d)

    # If the direction is not a descent direction, return 0 step size.
    if armijo_rhs_base >= 0:
        # Add a small tolerance for numerical stability
        if armijo_rhs_base > 1e-9:
           # print(f"Warning: Not a descent direction. <g, d> = {armijo_rhs_base}")
           return 0.0
        # If it's close to zero, we might still proceed if the objective decreases,
        # otherwise we treat it as 0.

    while True:
        x_new = x + eta * d
        f_new = objective_fn(x_new, A, y, loss_params)

        if f_new <= f_x + eta * armijo_rhs_base:
            return eta

        eta *= beta
        if eta < 1e-9:
            # Step size too small, stop iteration
            return eta

# --- 4.2 Exact IRLS / Newton's Method (Unified) ---

def irls(A, y, loss_type, loss_params, max_iters=30, tol=-1):
    n, d = A.shape
    x = np.zeros(d)
    history = []

    # Retrieve specific loss functions
    loss_fns = LOSS_FUNCTIONS[loss_type]
    gradient_fn = loss_fns['gradient']
    hessian_weights_fn = loss_fns['hessian_weights_scaled']
    objective_fn = loss_fns['objective']

    for t in range(max_iters):
        g = gradient_fn(x, A, y, loss_params)
        grad_norm = np.linalg.norm(g)
        obj_val = objective_fn(x, A, y, loss_params)
        # Record objective, gradient norm, and resketch count (which is t+1 for IRLS)
        history.append((obj_val, grad_norm, t+1))

        if tol >= 0 and grad_norm < tol:
            break

        # Calculate Hessian H = B.T @ B where B is the weighted A
        W_sqrt_scaled = hessian_weights_fn(x, A, y, loss_params)
        B = (A.T * W_sqrt_scaled).T
        H = B.T @ B

        # Solve H d = -g
        try:
            # Add small regularization for numerical stability, especially for logistic regression
            H_reg = H + np.eye(d) * 1e-8
            d_step = -np.linalg.solve(H_reg, g)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if solve fails
            d_step = -np.linalg.lstsq(H, g, rcond=None)[0]

        eta = armijo_backtracking(x, d_step, g, A, y, loss_fns, loss_params)

        if eta == 0:
            # print(f"IRLS stopped at iteration {t} due to zero step size.")
            break

        x = x + eta * d_step

    return x, history

# --- 4.3 Adaptive Lazy-IHS (Unified) ---

def adaptive_lazy_ihs(A, y, loss_type, loss_params, m, sketch_type, eta_min=0.1, max_iters=30, tol=-1, k_sparse=None):
    n, d = A.shape
    x = np.zeros(d)
    history = []
    resketch_count = 0
    t = 0
    resketch = True
    P_epoch = None

    # Retrieve specific functions
    loss_fns = LOSS_FUNCTIONS[loss_type]
    gradient_fn = loss_fns['gradient']
    hessian_weights_fn = loss_fns['hessian_weights_scaled']
    objective_fn = loss_fns['objective']
    sketch_fn = SKETCH_FUNCTIONS[sketch_type]

    # Pre-simulate S for specific variants (Sparse sketch fixed across epoch)
    S_L = None
    if sketch_type == 'sparse':
        S_L = sparse_random_projection_matrix(n, m)
    elif sketch_type == 'sparse_coordinate':
        if k_sparse is None:
             raise ValueError("k_sparse must be specified for 'sparse_coordinate' sketch.")
        # For sparse_coordinate, the sketch matrix for the remaining coordinates
        # can be precomputed if we assume the top-k indices don't change within an epoch.
        # However, since the top-k indices depend on B, which depends on x,
        # it's generally safer and more accurate to recompute the sketch entirely
        # at each resketch point, as the definition of "top-k" might change.
        # We will not precompute S_L for sparse_coordinate in this implementation
        # to prioritize correctness based on the current iterate x.
        pass


    while t < max_iters:
        g = gradient_fn(x, A, y, loss_params)
        grad_norm = np.linalg.norm(g)
        obj_val = objective_fn(x, A, y, loss_params)
        history.append((obj_val, grad_norm, resketch_count))

        if tol >= 0 and grad_norm < tol:
            break

        if resketch:
            resketch_count += 1

            # Form B = W_sqrt_scaled @ A
            W_sqrt_scaled = hessian_weights_fn(x, A, y, loss_params)
            B = (A.T * W_sqrt_scaled).T

            # Apply Sketch: SB = S @ B
            if sketch_type == 'sparse':
                # Pass the precomputed S_L if available
                SB = sketch_fn(B, m, S_precomputed=S_L)
            elif sketch_type == 'sparse_coordinate':
                # Pass k for the sparse coordinate sketch
                SB = sketch_fn(B, m, k=k_sparse)
            else:
                SB = sketch_fn(B, m)

            # Sketched Hessian H_hat = (SB).T @ (SB)
            H_hat = SB.T @ SB

            # Preconditioner P = pinv(H_hat)
            try:
                # Add small regularization for stability
                H_hat_reg = H_hat + np.eye(d) * 1e-8
                # Use a slightly higher rcond for pinv when m is small relative to d
                rcond = 1e-8 if m > d else 1e-6
                P_epoch = np.linalg.pinv(H_hat_reg, rcond=rcond)
            except np.linalg.LinAlgError:
                # Fallback if pinv fails (highly unlikely with regularization)
                P_epoch = np.identity(d)

            resketch = False

        # Newton step using the preconditioner: d = -P @ g
        d_step = -P_epoch @ g

        # Line search
        eta = armijo_backtracking(x, d_step, g, A, y, loss_fns, loss_params)

        # Check if line search failed
        if eta == 0:
            if not resketch:
                # If we haven't resketched yet, try resketching before giving up
                # print(f"Step size 0 at iteration {t}. Forcing resketch.")
                resketch = True
                continue
            else:
                # If we already resketched and still got eta=0, stop.
                # print(f"IRLS stopped at iteration {t} due to zero step size after resketch.")
                break

        x = x + eta * d_step
        t += 1

        # Adaptive condition: if step size is small, resketch
        # eta < 1.0 ensures we don't resketch if full Newton step was taken
        if eta < eta_min and eta < 1.0:
            resketch = True

    return x, history

# =============================================================================
# 5. Experiment Configuration and Execution
# =============================================================================

def run_experiment(config):
    """Runs the standard suite of experiments (IRLS, IHS K=1, Lazy-IHS with various sketches)."""
    print(f"\n=== Running Experiment Suite: {config['name']} ===")

    # --- Setup Parameters ---
    n = config['n']
    d = config['d']
    loss_type = config['loss_type']
    loss_params = config['loss_params']
    data_params = config.get('data_params', {})
    data_type = config.get('data_type', loss_type) # Default to loss_type if data_type not specified
    m_ratio = config.get('m_ratio', 3)
    m = m_ratio * d
    eta_min_adaptive = config.get('eta_min_adaptive', 0.2)
    max_iters = config.get('max_iters', 30)
    tol = config.get('tol', -1)
    # Parameter for sparse_coordinate sketch
    k_sparse_ratio = config.get('k_sparse_ratio', 0.1) # Ratio of m to keep isolated
    k_sparse = int(k_sparse_ratio * m)


    print(f"--- Configuration ---")
    print(f"Loss: {loss_type}, Data: {data_type}, n={n}, d={d}, m={m}")
    print(f"Data Params: {data_params}")
    print(f"Loss Params: {loss_params}")
    print(f"Adaptive eta_min={eta_min_adaptive}, Max Iters={max_iters}")
    print(f"Sparse Coordinate k={k_sparse} (Ratio={k_sparse_ratio})")


    # --- Data Generation ---
    data_generator = DATA_GENERATORS[data_type]
    # Generate data once for this configuration
    A, y, x_true = data_generator(n, d, data_params)

    # Update n and d in case the dataset loader returned different dimensions (e.g., YearPredictionMSD)
    n, d = A.shape
    m = min(m_ratio * d, n) # Ensure m <= n

    results = {}

    # --- Run Baseline IRLS (Exact Newton) ---
    # This is crucial to establish the optimal value (opt_val)
    print("\nRunning Baseline IRLS (Exact Newton)...")
    start_time = time.time()
    x_irls, history_irls = irls(A, y, loss_type, loss_params, max_iters=max_iters, tol=tol)
    opt_val = history_irls[-1][0]
    print(f"IRLS finished in {len(history_irls)} iterations. Time: {time.time()-start_time:.2f}s. Optimal Value: {opt_val:.6f}")
    results["IRLS (Exact)"] = history_irls

    # --- Define Lazy-IHS helper ---
    def lazy_ihs(A, y, m, sketch_type, eta_min, k_sparse=None):
         return adaptive_lazy_ihs(A, y, loss_type, loss_params, m, sketch_type=sketch_type, eta_min=eta_min, max_iters=max_iters, tol=tol, k_sparse=k_sparse)

    # --- Run Experiments with different Sketches ---
    # K=1 is achieved by setting eta_min > 1.0.
    experiments_to_run = [
        # (Name, Sketch Type, eta_min, k_sparse)
        ("IHS (K=1, Gauss)", 'gaussian', 1.1, None),
        ("Lazy-IHS (Adaptive, Gauss)", 'gaussian', eta_min_adaptive, None),
        ("Lazy-IHS (Adaptive, Sparse)", 'sparse', eta_min_adaptive, None),
        ("Lazy-IHS (Adaptive, SRHT)", 'srht', eta_min_adaptive, None),
        # Add the new sparse coordinate sketch experiment
        (f"Lazy-IHS (Adaptive, SparseCoord k={k_sparse})", 'sparse_coordinate', eta_min_adaptive, k_sparse),
    ]

    for name, sketch_type, eta_min, k_val in experiments_to_run:
        print(f"\nRunning {name}...")
        start_time = time.time()
        # Use the same data (A, y) for all runs within this configuration
        x_ihs, history_ihs = lazy_ihs(A, y, m, sketch_type, eta_min, k_sparse=k_val)
        # Ensure history is not empty before accessing the last element
        if history_ihs:
            resketches = history_ihs[-1][2]
        else:
            resketches = 0
        print(f"{name} finished in {len(history_ihs)} iterations. Resketches: {resketches}. Time: {time.time()-start_time:.2f}s.")
        results[name] = history_ihs

    return results, opt_val

def run_sketch_size_experiment(config, sketch_ratios):
    """Runs Lazy-IHS with varying sketch sizes (m) for a specific configuration."""
    print(f"\n=== Running Sketch Size Experiment: {config['name']} ===")

    # --- Setup Parameters ---
    n = config['n']
    d = config['d']
    loss_type = config['loss_type']
    loss_params = config['loss_params']
    data_params = config.get('data_params', {})
    data_type = config.get('data_type', loss_type)
    eta_min_adaptive = config.get('eta_min_adaptive', 0.2)
    max_iters = config.get('max_iters', 30)
    tol = config.get('tol', -1)
    sketch_type = 'gaussian' # Fixed sketch type for this experiment

    print(f"--- Configuration ---")
    print(f"Loss: {loss_type} ({sketch_type} sketch), n={n}, d={d}")
    print(f"Sketch Ratios (m/d): {sketch_ratios}")
    print(f"Adaptive eta_min={eta_min_adaptive}, Max Iters={max_iters}")

    # --- Data Generation ---
    data_generator = DATA_GENERATORS[data_type]
    A, y, x_true = data_generator(n, d, data_params)

     # Update n and d in case the dataset loader returned different dimensions
    n, d = A.shape

    results = {}

    # --- Run Baseline IRLS (Exact Newton) ---
    # Crucial for establishing opt_val
    print("\nRunning Baseline IRLS (Exact Newton) to find opt_val...")
    start_time = time.time()
    # Run IRLS for potentially more iterations to ensure good opt_val
    x_irls, history_irls = irls(A, y, loss_type, loss_params, max_iters=max(max_iters, 50), tol=tol)
    opt_val = history_irls[-1][0]
    print(f"IRLS finished. Optimal Value: {opt_val:.6f}")
    results["IRLS (Exact)"] = history_irls[:max_iters+1] # Trim history to match max_iters of others


    # --- Run Lazy-IHS with varying sketch sizes ---
    for ratio in sketch_ratios:
        m = int(ratio * d)
        # Ensure m is at least d for stability, and not larger than n
        m = max(d, min(m, n))
        name = f"Lazy-IHS (Gauss, m={m} [{ratio}d])"

        print(f"\nRunning {name}...")
        start_time = time.time()

        x_ihs, history_ihs = adaptive_lazy_ihs(
            A, y, loss_type, loss_params, m, sketch_type,
            eta_min=eta_min_adaptive, max_iters=max_iters, tol=tol
        )

        if history_ihs:
            resketches = history_ihs[-1][2]
        else:
            resketches = 0
        print(f"{name} finished in {len(history_ihs)} iterations. Resketches: {resketches}. Time: {time.time()-start_time:.2f}s.")
        results[name] = history_ihs


    return results, opt_val


# =============================================================================
# 6. Plotting and Analysis
# =============================================================================

def plot_results(results, opt_val, title="Experiment Results", save_dir=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.set_style("whitegrid")

    # Define colors and styles for better differentiation
    color_palette = sns.color_palette("tab10", len(results))
    styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]

    # Assign styles and colors
    method_styles = {}
    i = 0
    for name in results.keys():
        method_styles[name] = {'color': color_palette[i % len(color_palette)],
                               'linestyle': styles[i % len(styles)]}
        i += 1

    # --- Plot 1: Convergence Rate (Iterations) ---
    ax1 = axes[0]
    for name, history in results.items():
        if not history: continue # Skip if history is empty
        iterations = range(len(history))
        # Calculate suboptimality F(x_t) - F(x*)
        suboptimality = [h[0] - opt_val for h in history]
        # Ensure non-negative values for log scale plotting
        suboptimality = [max(s, 1e-10) for s in suboptimality]
        style = method_styles[name]
        ax1.plot(iterations, suboptimality, label=name, linewidth=2, **style)

    ax1.set_yscale('log')
    ax1.set_xlabel('Iteration (t)')
    ax1.set_ylabel('Suboptimality $F(x_t) - F(x^*)$')
    ax1.set_title(f'{title}: Iterations')
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    # --- Plot 2: Computational Complexity (Sketching Cost) ---
    ax2 = axes[1]
    for name, history in results.items():
        if not history: continue

        suboptimality = [max(h[0] - opt_val, 1e-10) for h in history]
        # The third element in history is the cumulative sketch count
        cumulative_sketches = [h[2] for h in history]

        # We want to plot the suboptimality achieved *after* a certain number of sketches.
        # This involves filtering the history to points where the sketch count increases.
        sketch_counts = []
        sketch_subopt = []
        last_count = -1

        for i in range(len(cumulative_sketches)):
            count = cumulative_sketches[i]
            if count > last_count:
                sketch_counts.append(count)
                # Plot the suboptimality achieved by the iteration that used this sketch count
                sketch_subopt.append(suboptimality[i])
                last_count = count

        if sketch_counts:
            style = method_styles[name]
            ax2.plot(sketch_counts, sketch_subopt, label=name, linewidth=2, marker='o', markersize=5, **style)

    ax2.set_yscale('log')
    ax2.set_xlabel('Cumulative Hessian Sketches/Evaluations ($N_R$)')
    ax2.set_ylabel('Suboptimality $F(x_t) - F(x^*)$')
    ax2.set_title(f'{title}: Sketching Cost')

    # Create a single legend outside the plots
    handles, labels = ax1.get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make room for the legend
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')

    ax2.grid(True, which="both", ls="--", alpha=0.5)

    # Save the figure if save_dir is provided and exists
    if save_dir and os.path.exists(save_dir):
        try:
            # Sanitize title for filename
            filename_title = "".join([c if c.isalnum() or c in (' ', '-') else '_' for c in title]).rstrip()
            filename = f"{filename_title.replace(' ', '_')}_results.png"
            save_path = os.path.join(save_dir, filename)
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Plot saved to: {save_path}")
        except Exception as e:
            print(f"Error saving plot {title}: {e}")


    plt.show()

    # --- Summary Statistics ---
    print(f"\n--- Summary Statistics ({title}) ---")
    summary_data = []
    for name, history in results.items():
        if not history:
            print(f"Warning: No history for {name}")
            continue

        total_iters = len(history)
        final_subopt = max(history[-1][0] - opt_val, 0)
        # The total sketches are recorded in the last history entry
        total_sketches = history[-1][2]

        # Calculate average amortization K = Total Iterations / Total Sketches
        if total_sketches > 0:
            avg_amortization = total_iters / total_sketches
            avg_amortization_str = f"{avg_amortization:.2f}"
        else:
            # Handle case with 0 sketches (e.g., if it converged instantly)
            avg_amortization_str = "N/A"


        summary_data.append({
            "Method": name,
            "Iterations": total_iters,
            "Sketches (N_R)": total_sketches,
            "Avg. Amortization (K)": avg_amortization_str,
            "Final Suboptimality": f"{final_subopt:.2e}"
        })

    df_summary = pd.DataFrame(summary_data)
    # Use print(df_summary) if to_markdown is not available or desired
    try:
        print(df_summary.to_markdown(index=False))
    except ImportError:
        print(df_summary)


if __name__ == "__main__":

    np.random.seed(42)

    plot_save_dir = './plots'

    data_file = 'YearPredictionMSD_10k.csv'
    df_10k = pd.read_csv(data_file)

    y_year = df_10k['year'].values
    A_year = df_10k.drop('year', axis=1).values

    print("loaded dataframe")

    print("\nShape of feature matrix A_year:", A_year.shape)
    print("Shape of target vector y_year:", y_year.shape)

    # --- Configuration 1: Pseudo-Huber Robust Regression (Standard Suite) ---
    config_huber = {
            'name': 'Pseudo-Huber Regression (Synthetic)',
            'n': 10000,
            'd': 200,
            'loss_type': 'pseudo_huber',
            'data_type': 'pseudo_huber',
            'loss_params': {'sigma': 1.0},
            'data_params': {'outlier_fraction': 0.2, 'outlier_magnitude': 50},
            'm_ratio': 3,
            'k_sparse_ratio': 0.2, # 20% of the sketch dedicated to top coordinates
            'eta_min_adaptive': 0.2,
            'max_iters': 30,
    }

    # Run standard experiment suite
    start_exp_time = time.time()
    results_huber, opt_val_huber = run_experiment(config_huber)
    print(f"\nTotal time for {config_huber['name']}: {time.time()-start_exp_time:.2f}s")
    plot_results(results_huber, opt_val_huber, title=config_huber['name'], save_dir=plot_save_dir)
    print("-" * 60)


    # --- Configuration 2: Pseudo-Huber Varying Sketch Size ---
    config_huber_vary_m = config_huber.copy()
    config_huber_vary_m['name'] = 'Pseudo-Huber (Varying Sketch Size)'
    # Define the ratios m/d to test
    sketch_ratios = [1.5, 3, 5, 10]

    # Run sketch size experiment
    start_exp_time = time.time()
    results_vary_m, opt_val_vary_m = run_sketch_size_experiment(config_huber_vary_m, sketch_ratios)
    print(f"\nTotal time for {config_huber_vary_m['name']}: {time.time()-start_exp_time:.2f}s")
    plot_results(results_vary_m, opt_val_vary_m, title=config_huber_vary_m['name'], save_dir=plot_save_dir)
    print("-" * 60)

    # --- Configuration 3: Logistic Regression (Standard Suite) ---
    config_logistic = {
            'name': 'Logistic Regression (Synthetic)',
            'n': 10000,
            'd': 200,
            'loss_type': 'logistic',
            'data_type': 'logistic',
            'loss_params': {},
            'data_params': {'outlier_fraction': 0.2},
            'm_ratio': 3,
            'k_sparse_ratio': 0.2,
            'eta_min_adaptive': 0.35,
            'max_iters': 20,
        }

    # Run standard experiment suite
    start_exp_time = time.time()
    results_logistic, opt_val_logistic = run_experiment(config_logistic)
    print(f"\nTotal time for {config_logistic['name']}: {time.time()-start_exp_time:.2f}s")
    plot_results(results_logistic, opt_val_logistic, title=config_logistic['name'], save_dir=plot_save_dir)
    print("-" * 60)

    # --- Configuration 4: YearPredictionMSD Dataset with Pseudo-Huber Loss ---

    # Check if the dataset was loaded successfully
    if A_year is not None and y_year is not None:
        # Use the actual shape from the loaded data
        n_msd, d_msd = A_year.shape

        config_msd_huber = {
            'name': 'YearPredictionMSD (Pseudo-Huber)',
            'n': n_msd, # Use actual n
            'd': d_msd, # Use actual d
            'loss_type': 'pseudo_huber',
            'data_type': 'year_prediction_msd', # Use the specific data generator
            'loss_params': {'sigma': 10.0}, # Adjust sigma based on the scale of the target variable
            'data_params': {},
            'm_ratio': 5, # Sketch ratio m/d
            'k_sparse_ratio': 0.25, # Dedicate 25% of the sketch to top coordinates
            'eta_min_adaptive': 0.2,
            'max_iters': 40,
        }

        # Run the specified experiment configuration
        start_exp_time = time.time()
        results_msd_huber, opt_val_msd_huber = run_experiment(config_msd_huber)
        print(f"\nTotal time for {config_msd_huber['name']}: {time.time()-start_exp_time:.2f}s")
        plot_results(results_msd_huber, opt_val_msd_huber, title=config_msd_huber['name'], save_dir=plot_save_dir)
        print("-" * 60)
    else:
        print("\nSkipping YearPredictionMSD experiment because the data failed to load.")