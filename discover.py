"""
PC with hard background-knowledge constraints.

causal-learn's BackgroundKnowledge handles forbidden edges in the skeleton phase,
but a *required* edge only orients an adjacency that survives the independence
tests; PC can still delete it. To make required edges hard constraints, we wrap
the conditional-independence test so that any required pair always looks
dependent (p = 0). PC then never finds a separating set for that pair, so the
adjacency is kept, and BackgroundKnowledge orients it as claimed.

This mirrors the LLM + PC hybrid of Srivastava et al. (2025): prior edges are
never removed in skeleton discovery and are oriented before Meek's rules run.

Usage
-----
    cg = run_pc(X, names, required=[("X1", "X2")], forbidden=[("X3", "X7")])
    cg.G.graph   # causal-learn adjacency matrix: graph[j, i] == 1 and graph[i, j] == -1  means  i -> j
"""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
from causallearn.graph.GraphNode import GraphNode
from causallearn.utils.cit import CIT
from causallearn.utils.PCUtils import Meek, SkeletonDiscovery, UCSepset
from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
from causallearn.utils.PCUtils.BackgroundKnowledgeOrientUtils import orient_by_background_knowledge

Edge = Tuple[str, str]


class ForceKeepCIT:
    """Wraps a causal-learn CIT so that protected pairs always test as dependent.

    `protected` holds unordered pairs of column indices. For any test X ⫫ Y | S
    where {X, Y} is protected, the wrapper returns p = 0.0 regardless of S, so PC
    never removes that adjacency. All other tests pass through unchanged.
    """

    def __init__(self, inner_cit, protected: Iterable[Tuple[int, int]]):
        self.inner = inner_cit
        self.protected = {frozenset(p) for p in protected}

    def __call__(self, x, y, condition_set=None):
        if frozenset((x, y)) in self.protected:
            return 0.0
        return self.inner(x, y, condition_set)

    def __getattr__(self, name):
        # Delegate everything else (method, data, cache, ...) to the wrapped test.
        # Guard: during copy/deepcopy the object exists before `inner` is set, and
        # looking up self.inner here would recurse forever.
        if name == "inner":
            raise AttributeError(name)
        return getattr(self.inner, name)


def build_background_knowledge(names: Sequence[str],
                               required: Iterable[Edge] = (),
                               forbidden: Iterable[Edge] = ()) -> BackgroundKnowledge:
    """Required edges are directional (X -> Y). Forbidden edges remove the
    adjacency entirely, so both directions are forbidden."""
    nodes = {n: GraphNode(n) for n in names}
    bk = BackgroundKnowledge()
    for a, b in required:
        bk.add_required_by_node(nodes[a], nodes[b])
    for a, b in forbidden:
        bk.add_forbidden_by_node(nodes[a], nodes[b])
        bk.add_forbidden_by_node(nodes[b], nodes[a])
    return bk


def run_pc(X: np.ndarray,
           names: Sequence[str],
           required: Iterable[Edge] = (),
           forbidden: Iterable[Edge] = (),
           alpha: float = 0.05,
           test: str = "fisherz"):
    """Run PC where every required edge and every forbidden edge is a hard constraint.

    Parameters
    ----------
    X         : (n_samples, n_vars) data, columns in the same order as `names`
    names     : variable names
    required  : (a, b) pairs meaning "a -> b exists"; never removed, oriented a -> b
    forbidden : (a, b) pairs meaning "no edge between a and b"; always removed
    alpha     : significance level for the CI tests
    test      : causal-learn CI test name ("fisherz" for continuous, "chisq" for discrete)

    Returns
    -------
    causal-learn CausalGraph (CPDAG, with background-knowledge orientations applied)
    """
    required, forbidden = list(required), list(forbidden)
    names = list(names)
    idx = {n: i for i, n in enumerate(names)}

    clash = {frozenset(e) for e in required} & {frozenset(e) for e in forbidden}
    if clash:
        raise ValueError(f"Pairs both required and forbidden: {[tuple(c) for c in clash]}")
    req_pairs = {frozenset(e) for e in required}
    if len(req_pairs) < len(required):
        raise ValueError("A pair appears twice in `required` (possibly in both directions).")

    bk = build_background_knowledge(names, required, forbidden)
    cit = ForceKeepCIT(CIT(X, test), [(idx[a], idx[b]) for a, b in required])

    # Same steps and defaults as causallearn's pc() (stable=True, uc_rule=0,
    # uc_priority=2 = prioritize existing colliders), but with our wrapped test.
    cg = SkeletonDiscovery.skeleton_discovery(X, alpha, cit, stable=True,
                                              background_knowledge=bk,
                                              show_progress=False, node_names=names)
    orient_by_background_knowledge(cg, bk)
    cg = UCSepset.uc_sepset(cg, 2, background_knowledge=bk)
    cg = Meek.meek(cg, background_knowledge=bk)
    return cg


def edge_state(cg, names: Sequence[str], a: str, b: str) -> str:
    """Return '->', '<-', '--', '<->' or 'none' for the pair (a, b) in the learned graph."""
    i, j = names.index(a), names.index(b)
    g = cg.G.graph
    if g[i, j] == 0 and g[j, i] == 0:
        return "none"
    if g[j, i] == 1 and g[i, j] == -1:
        return "->"
    if g[i, j] == 1 and g[j, i] == -1:
        return "<-"
    if g[i, j] == -1 and g[j, i] == -1:
        return "--"
    return "<->"