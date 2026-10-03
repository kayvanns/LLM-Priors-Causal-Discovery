"""
Background knowledge ("hints") for PC: build a correct set, then corrupt it.

    K = correct_knowledge(dag, coverage=0.75, rng=random.Random(seed))
    bad = corrupt(K, dag, "false_positive", e=0.2, rng=random.Random(f"{seed}-false_positive"))
    run_pc(X, dag["names"], required=bad["required"], forbidden=bad["forbidden"])

or in one call:

    hints = make_knowledge(dag, coverage=0.75, seed=0, error_type="false_negative", e=0.3)

Error types (one at a time, a fraction e of the relevant claims):
    false_positive   replace e of the REQUIRED claims with arrows between pairs that are
                     not linked in truth (and not already forbidden)
    false_negative   replace e of the FORBIDDEN claims with pairs that ARE linked in truth;
                     drawn first from true edges not in required, then from required edges
                     (which are then removed from required)
    wrong_direction  reverse e of the REQUIRED claims

Errors nest: called with a fresh rng from the same (seed, error_type), the claims corrupted
at e = 0.2 are always among those corrupted at e = 0.4. Each step up only adds errors.
"""
import random

from graphs import load_graph, nonadjacent_pairs

ERROR_TYPES = ["false_positive", "false_negative", "wrong_direction"]

def shuffled(items, rng):
    """A shuffled copy. Shuffling once and taking the first k is what makes errors nest."""
    out = list(items)
    rng.shuffle(out)
    return out


def correct_knowledge(dag, coverage, rng):
    """Required: a `coverage` share of true edges (true direction).
    Forbidden: the same number of truly non-adjacent pairs."""
    edges = list(dag["edges"])
    n = round(coverage * len(edges))
    required = shuffled(dag["edges"], rng)[:n]
    forbidden = shuffled(nonadjacent_pairs(dag), rng)[:n]
    return {"required": required, "forbidden": forbidden}




def corrupt(K, dag, error_type, e, rng):
    """Return a corrupted COPY of K; K itself is never changed."""
    required = list(K["required"])
    forbidden = list(K["forbidden"])

    if error_type == "wrong_direction":
        k = round(e * len(required))
        flip = set(shuffled(required, rng)[:k])
        required = [(b, a) if (a, b) in flip else (a, b) for (a, b) in required]

    elif error_type == "false_positive":
        k = round(e * len(required))
        drop = set(shuffled(required, rng)[:k])

        # Fake arrows: pairs not linked in truth and not already forbidden (avoids clashes)
        forbidden_pairs = {frozenset(p) for p in forbidden}
        candidates = [p for p in nonadjacent_pairs(dag) if frozenset(p) not in forbidden_pairs]
        candidates = shuffled(candidates, rng)
        flips = [rng.random() < 0.5 for _ in candidates]          # random direction for each fake arrow
        fake = [(b, a) if f else (a, b) for (a, b), f in zip(candidates, flips)][:k]

        required = [c for c in required if c not in drop] + fake

    elif error_type == "false_negative":
        k = round(e * len(forbidden))
        drop = set(shuffled(forbidden, rng)[:k])

        # Real edges to forbid: first those not already required, then required ones
        required_set = set(required)
        spare = shuffled([edge for edge in dag["edges"] if edge not in required_set], rng)
        from_required = shuffled(required, rng)
        real = (spare + from_required)[:k]

        forbidden = [c for c in forbidden if c not in drop] + real
        taken = set(real)
        required = [c for c in required if c not in taken]     # can't be required and forbidden

    else:
        raise ValueError(f"unknown error_type {error_type!r}; expected one of {ERROR_TYPES}")

    return {"required": required, "forbidden": forbidden}


def make_knowledge(dag, coverage, seed, error_type=None, e=0.0):
    """Correct hints for this seed, corrupted by `error_type` at rate `e`.
 
    The correct set depends only on the seed (and coverage, with smaller coverages nested
    inside larger ones), so every error type and rate starts from the same hints. The
    corruption depends only on (seed, error_type), so errors nest across rates.
    """
    K = correct_knowledge(dag, coverage, random.Random(f"{seed}-K"))
    if error_type is None or e == 0:
        return K
    return corrupt(K, dag, error_type, e, random.Random(f"{seed}-{error_type}"))


def count_errors(hints, dag):
    """How many claims are wrong, by kind. Used to check corrupt()."""
    true_edges = set(dag["edges"])
    adjacent = {frozenset(edge) for edge in dag["edges"]}
    req = hints["required"]
    return {
        "fake_arrows": sum(frozenset(c) not in adjacent for c in req),
        "reversed": sum((b, a) in true_edges for (a, b) in req),
        "forbidden_real_edges": sum(frozenset(c) in adjacent for c in hints["forbidden"]),
    }


if __name__ == "__main__":
    dag = load_graph("covid_respiratory")
    K = make_knowledge(dag, 0.75, seed=0)
    print("correct:", len(K["required"]), "required,", len(K["forbidden"]), "forbidden,",
          "errors:", count_errors(K, dag))
    for t in ERROR_TYPES:
        for e in (0.2, 0.5):
            h = make_knowledge(dag, 0.75, seed=0, error_type=t, e=e)
            print(f"{t:16s} e={e}: {len(h['required'])} required, {len(h['forbidden'])} forbidden, "
                  f"errors {count_errors(h, dag)}")