"""
The ground-truth causal graphs: the answer key.


A graph is stored as a plain dictionary:

    dag = load_graph("covid_respiratory")
    dag["names"]    # list of variable names (this is also the column order of every data matrix)
    dag["edges"]    # list of true (cause, effect) pairs
    dag["G"]        # the same graph as a networkx DiGraph, for path questions
    dag["index"]    # {name: column number}
    dag["order"]    # names in causal order (every cause before its effects)

Main functions:

    confounded_pairs(dag)                                  # (T, Y) pairs that need adjustment
    is_valid_adjustment(dag, "tau", "moca", {"age", "av45"})  # backdoor criterion in the TRUE graph
    true_total_effect(dag, B, "tau", "moca")               # exact causal effect for weights B

Backdoor criterion (Pearl): adjusting for a set Z gives the true effect of T on Y if
  (1) nothing in Z is caused by T (no descendants of T), and
  (2) Z blocks every path between T and Y that starts with an arrow INTO T.
A "confounded pair" is a (T, Y) pair, with Y downstream of T, where adjusting for
nothing is NOT enough. Pairs that need no adjustment are left out of the main metric,
because any learned graph would pass them.
"""
import json
from itertools import permutations
from pathlib import Path

import networkx as nx
import numpy as np

GRAPH_DIR = Path(__file__).resolve().parent.parent / "graphs"
GRAPHS = ["alzheimers", "covid_respiratory"]


def load_graph(name):
    """Read data/graphs/<name>.json and return the graph as a dictionary."""
    with open(GRAPH_DIR / f"{name}.json") as f:
        raw = json.load(f)

    names = list(raw["nodes"].values())
    edges = [(a, b) for a, b in raw["edges"]]

    G = nx.DiGraph()
    G.add_nodes_from(names)
    G.add_edges_from(edges)

    # Basic safety checks on the file
    unknown = {v for e in edges for v in e} - set(names)
    if unknown:
        raise ValueError(f"{name}: edges mention unknown nodes {unknown}")
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError(f"{name}: graph has a cycle")

    return {
        "name": name,
        "names": names,
        "edges": edges,
        "descriptions": raw["nodes"],
        # "G": G,
        "index": {v: i for i, v in enumerate(names)},
        "order": list(nx.topological_sort(G)),
    }


# ---------- simple structure questions ----------

def parents(dag, v):
    return set(dag["G"].predecessors(v))


def descendants(dag, v):
    """Everything downstream of v (caused by v directly or indirectly)."""
    return nx.descendants(dag["G"], v)


def nonadjacent_pairs(dag):
    """Pairs (a, b) with no edge in either direction, in column order."""
    adjacent = {frozenset(e) for e in dag["edges"]}
    names = dag["names"]
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if frozenset((names[i], names[j])) not in adjacent:
                pairs.append((names[i], names[j]))
    return pairs


# ---------- adjustment sets ----------

def is_valid_adjustment(dag, T, Y, Z):
    """Is Z a valid adjustment set for the effect of T on Y in the TRUE graph?"""
    Z = set(Z)
    if T in Z or Y in Z:
        return False

    # (1) Nothing in Z may be caused by T
    if Z & descendants(dag, T):
        return False

    # (2) Delete T's outgoing arrows; whatever still connects T and Y is a backdoor path.
    #     Z is valid if it blocks (d-separates) all of them.
    g = dag["G"].copy()
    g.remove_edges_from(list(g.out_edges(T)))
    return nx.is_d_separator(g, {T}, {Y}, Z)


def confounded_pairs(dag):
    """All (T, Y) with Y downstream of T where adjusting for nothing is NOT valid."""
    pairs = []
    for T, Y in permutations(dag["names"], 2):
        if Y in descendants(dag, T) and not is_valid_adjustment(dag, T, Y, set()):
            pairs.append((T, Y))
    return pairs


# ---------- exact effects in a linear SEM ----------

def true_total_effects(dag, B):
    """Matrix of exact total effects: entry [i, j] = effect of variable i on variable j.

    B[i, j] is the weight of the edge i -> j (0 if there is no edge). The total effect
    sums, over every directed path from i to j, the product of the weights on that path.
    B covers one-step paths, B @ B two-step paths, and so on; the sum
    B + B^2 + B^3 + ... equals (I - B)^-1 - I (like 1 + x + x^2 + ... = 1 / (1 - x)).
    """
    k = len(dag["names"])
    I = np.eye(k)
    return np.linalg.inv(I - B) - I


def true_total_effect(dag, B, T, Y):
    effects = true_total_effects(dag, B)
    return float(effects[dag["index"][T], dag["index"][Y]])


def summary(dag):
    return (f"{dag['name']}: {len(dag['names'])} nodes, {len(dag['edges'])} edges, "
            f"{len(confounded_pairs(dag))} confounded (T, Y) pairs")


if __name__ == "__main__":
    # for g in GRAPHS:
    #     print(summary(load_graph(g)))
    print(json.dumps(load_graph("covid_respiratory")))