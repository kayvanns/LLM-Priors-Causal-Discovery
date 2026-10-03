"""
How good is a learned graph? 
 
1. Graph level: compare the learned graph to the true graph, edge by edge.
 
    graph_metrics(cg, dag)
    -> {"shd": ..., "adj_precision": ..., "adj_recall": ..., "adj_f1": ...,
        "arrow_precision": ..., "arrow_recall": ..., "arrow_f1": ...}
 
2. Decision level: would an analyst using the learned graph reach the right causal
   conclusion? For every confounded pair (T, Y) in the true graph:
 
    adjustment_metrics(cg, dag, X, B)
    -> one row (dictionary) per pair, with:
       status    "ok" | "not_identified" | "reversed"
       valid     is the learned adjustment set valid in the TRUE graph?  (main metric)
       material  is the conclusion materially different from a correct analysis?
 
   Rule for "materially different": the effect estimate from the learned adjustment set
   falls outside the 95% confidence interval of the estimate from a correct adjustment
   set (the true parents of T), or the estimated effect has the opposite sign.
 
Learned-graph states for a pair (a, b), from discover.edge_state:
   "->"   a causes b          "<-"   b causes a
   "--"   adjacent, direction unknown
   "<->"  conflicting orientations      "none"  not adjacent
"""
import numpy as np
 
from discover import edge_state
from graphs import confounded_pairs, is_valid_adjustment, parents, true_total_effect
 
Z_95 = 1.959964   # normal quantile for a 95% interval

def true_state(dag, a, b):
    if (a, b) in dag["edges"]:
        return "->"
    if (b, a) in dag["edges"]:
        return "<-"
    return "none"
 
 
def safe_div(x, y):
    return x / y if y else 0.0
 
 
def f1(p, r):
    return safe_div(2 * p * r, p + r)
 
 
def graph_metrics(cg, dag):
    """Compare a learned graph (causal-learn output) to the true DAG."""
    names = dag["names"]
    shd = 0
    adj_tp = adj_fp = adj_fn = 0          # adjacencies (ignoring direction)
    arrow_tp = arrow_learned = 0          # directed edges in the learned graph
 
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            truth = true_state(dag, a, b)
            learned = edge_state(cg, names, a, b)
 
            # SHD: one point for every pair that is not exactly right
            # (missing, extra, reversed, or left undirected)
            if learned != truth:
                shd += 1
 
            if truth != "none" and learned != "none":
                adj_tp += 1
            elif truth == "none" and learned != "none":
                adj_fp += 1
            elif truth != "none" and learned == "none":
                adj_fn += 1
 
            if learned in ("->", "<-"):
                arrow_learned += 1
                if learned == truth:
                    arrow_tp += 1
 
    n_true_edges = len(dag["edges"])
    adj_p, adj_r = safe_div(adj_tp, adj_tp + adj_fp), safe_div(adj_tp, adj_tp + adj_fn)
    arr_p, arr_r = safe_div(arrow_tp, arrow_learned), safe_div(arrow_tp, n_true_edges)
    return {
        "shd": shd,
        "adj_precision": adj_p, "adj_recall": adj_r, "adj_f1": f1(adj_p, adj_r),
        "arrow_precision": arr_p, "arrow_recall": arr_r, "arrow_f1": f1(arr_p, arr_r),
    }
 

def learned_adjustment_set(cg, dag, T):
    """Parents of T in the learned graph.
 
    Returns (Z, status). If any edge at T has no clear direction ("--" or "<->"),
    the learned graph does not say what T's parents are: status "not_identified".
    """
    names = dag["names"]
    Z = set()
    for v in names:
        if v == T:
            continue
        s = edge_state(cg, names, v, T)
        if s == "->":
            Z.add(v)
        elif s in ("--", "<->"):
            return None, "not_identified"
    return Z, "ok"
 
 
def ols_effect(X, dag, T, Y, Z):
    """OLS of Y on an intercept, T and Z. Returns (coefficient on T, its standard error)."""
    idx = dag["index"]
    cols = [idx[T]] + [idx[z] for z in sorted(Z)]
    design = np.column_stack([np.ones(len(X)), X[:, cols]])
    y = X[:, idx[Y]]
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ beta
    dof = len(y) - design.shape[1]
    sigma2 = resid @ resid / dof
    cov = sigma2 * np.linalg.inv(design.T @ design)
    return float(beta[1]), float(np.sqrt(cov[1, 1]))
 
 
def adjustment_metrics(cg, dag, X, B):
    """One row per confounded (T, Y) pair: is the learned adjustment set valid, and does
    the resulting effect estimate lead to a materially different conclusion?"""
    rows = []
    for T, Y in confounded_pairs(dag):
        # Reference: a correct adjustment set (true parents of T), same data
        ref_est, ref_se = ols_effect(X, dag, T, Y, parents(dag, T))
        ref_lo, ref_hi = ref_est - Z_95 * ref_se, ref_est + Z_95 * ref_se
 
        row = {
            "T": T, "Y": Y,
            "true_effect": true_total_effect(dag, B, T, Y),
            "ref_est": ref_est, "ref_lo": ref_lo, "ref_hi": ref_hi,
            "adj_set": None, "est": np.nan,
            "valid": False, "sign_flip": False, "material": True,
        }
 
        Z, status = learned_adjustment_set(cg, dag, T)
        if status == "ok" and Y in Z:
            status = "reversed"            # learned graph says Y causes T
        row["status"] = status
 
        if status == "ok":
            est, _ = ols_effect(X, dag, T, Y, Z)
            sign_flip = np.sign(est) != np.sign(ref_est)
            outside = est < ref_lo or est > ref_hi
            row.update({
                "adj_set": ",".join(sorted(Z)),
                "est": est,
                "valid": is_valid_adjustment(dag, T, Y, Z),
                "sign_flip": bool(sign_flip),
                "material": bool(sign_flip or outside),
            })
        rows.append(row)
    return rows
 
 
def summarize_adjustment(rows):
    """Shares across all pairs in one run (for quick printing)."""
    n = len(rows)
    return {
        "p_valid": sum(r["valid"] for r in rows) / n,
        "p_material": sum(r["material"] for r in rows) / n,
        "p_not_identified": sum(r["status"] == "not_identified" for r in rows) / n,
    }