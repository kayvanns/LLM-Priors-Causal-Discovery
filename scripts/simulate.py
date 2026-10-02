"""
Observational data from a true DAG via a linear Gaussian structural equation model (SEM).

Following Srivastava et al. (2025):

    X_j = sum_{i in Pa(j)} w_ij * X_i + eps_j,     w_ij ~ U(0, 2),   eps_j ~ N(0, 1)

Each seed draws NEW weights and NEW data, so results average over many plausible
"worlds" consistent with the same graph, not just one.

    from graphs import load_graph, true_total_effect
    from simulate import simulate

    dag = load_graph("covid_respiratory")
    sem = simulate(dag, n=2000, seed=0)
    sem["X"]     # (2000, 11) data matrix, columns in dag["names"] order
    sem["B"]     # (11, 11) weights: B[i, j] = weight of edge i -> j
    true_total_effect(dag, sem["B"], "X7", "X8")   # exact effect, for checking bias later
"""
import numpy as np

from graphs import load_graph

W_LOW, W_HIGH = 0.0, 2.0      # weight range (matches Srivastava et al.)
NOISE_SD = 1.0


def sample_weights(dag, rng, low=W_LOW, high=W_HIGH):
    """One weight per true edge, drawn uniformly from [low, high]. Returns the matrix B."""
    k = len(dag["names"])
    B = np.zeros((k, k))
    for a, b in dag["edges"]:
        B[dag["index"][a], dag["index"][b]] = rng.uniform(low, high)
    return B


def sample_data(dag, B, n, rng, noise_sd=NOISE_SD):
    """Generate n rows, visiting variables in causal order so every cause exists before its effects."""
    X = np.zeros((n, len(dag["names"])))
    for v in dag["order"]:
        j = dag["index"][v]
        X[:, j] = X @ B[:, j] + rng.normal(0.0, noise_sd, n)   # B[:, j] is zero except for v's parents
    return X


def simulate(dag, n, seed, low=W_LOW, high=W_HIGH):
    """Fresh weights and data for one seed. Returns {"X": data, "B": weights, "seed": seed}.

    The same (graph, seed) always gives the same weights, whatever n is, so different
    sample sizes describe the same underlying world.
    """
    w_rng = np.random.default_rng([seed, 0])          # weights depend on seed only
    d_rng = np.random.default_rng([seed, 1, n])       # data depends on seed and n
    B = sample_weights(dag, w_rng, low, high)
    X = sample_data(dag, B, n, d_rng)
    return {"X": X, "B": B, "seed": seed}


if __name__ == "__main__":
    for g in ("alzheimers", "covid_respiratory"):
        dag = load_graph(g)
        sem = simulate(dag, n=2000, seed=0)
        print(f"{g}: X shape {sem['X'].shape}, {int((sem['B'] != 0).sum())} nonzero weights, "
              f"column SDs {np.round(sem['X'].std(0), 1)}")