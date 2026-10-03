"""
Run the full experiment grid and save the results.

For every graph, sample size n and seed:
  1. simulate data once (the same data is used by every condition below)
  2. baseline:   PC with no background knowledge
  3. for each coverage (0.75 main, 0.5 comparison):
       correct:    PC with correct hints (e = 0)
       corrupted:  PC with hints corrupted by each error type at each e > 0

Outputs (in results/):
  graph_metrics.csv   one row per PC run: SHD, adjacency and arrowhead precision/recall/F1,
                      plus the share of confounded pairs that are valid / material / not identified
  adjustment.csv      one row per PC run x confounded (T, Y) pair: status, learned adjustment
                      set, valid, estimate, reference estimate and CI, true effect, material

Usage:
  python3 scripts/run_experiments.py                       # full grid
  python3 scripts/run_experiments.py --seeds 2 --n 1000    # quick test
  python3 scripts/run_experiments.py --jobs 4              # use 4 CPU cores
"""
import argparse
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from discover import run_pc
from graphs import GRAPHS, load_graph
from knowledge import ERROR_TYPES, make_knowledge
from metrics import adjustment_metrics, graph_metrics, summarize_adjustment
from simulate import simulate

SAMPLE_SIZES = [250, 1000, 5000]
N_SEEDS = 50
COVERAGES = [0.75, 0.5]
ERROR_RATES = [0.1, 0.2, 0.3, 0.4, 0.5]          # e = 0 is the "correct" condition

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def score_run(dag, sem, hints, labels):
    """Run PC with the given hints and score it. Returns (one graph row, list of pair rows)."""
    cg = run_pc(sem["X"], dag["names"], required=hints["required"], forbidden=hints["forbidden"])

    pair_rows = adjustment_metrics(cg, dag, sem["X"], sem["B"])
    graph_row = {**labels,
                 "n_required": len(hints["required"]),
                 "n_forbidden": len(hints["forbidden"]),
                 **graph_metrics(cg, dag),
                 **summarize_adjustment(pair_rows)}
    pair_rows = [{**labels, **row} for row in pair_rows]
    return graph_row, pair_rows


def run_one_dataset(task):
    """Every condition for one (graph, n, seed). Runs in its own process when --jobs > 1."""
    graph_name, n, seed, coverages, error_types, error_rates = task
    dag = load_graph(graph_name)
    sem = simulate(dag, n=n, seed=seed)
    base = {"graph": graph_name, "n": n, "seed": seed}

    graph_rows, pair_rows = [], []

    def record(hints, condition, coverage, error_type, e):
        labels = {**base, "condition": condition, "coverage": coverage,
                  "error_type": error_type, "e": e}
        g, p = score_run(dag, sem, hints, labels)
        graph_rows.append(g)
        pair_rows.extend(p)

    # 1. Baseline: no knowledge
    record({"required": [], "forbidden": []}, "baseline", 0.0, "none", 0.0)

    for coverage in coverages:
        # 2. Correct knowledge (e = 0), shared by all error types
        record(make_knowledge(dag, coverage, seed), "correct", coverage, "none", 0.0)

        # 3. Corrupted knowledge
        for error_type in error_types:
            for e in error_rates:
                hints = make_knowledge(dag, coverage, seed, error_type, e)
                record(hints, "corrupted", coverage, error_type, e)

    return graph_rows, pair_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", nargs="+", default=GRAPHS)
    parser.add_argument("--n", nargs="+", type=int, default=SAMPLE_SIZES)
    parser.add_argument("--seeds", type=int, default=N_SEEDS)
    parser.add_argument("--coverage", nargs="+", type=float, default=COVERAGES)
    parser.add_argument("--jobs", type=int, default=1, help="number of CPU cores to use")
    parser.add_argument("--out", default=str(RESULTS_DIR))
    args = parser.parse_args()

    tasks = [(g, n, seed, args.coverage, ERROR_TYPES, ERROR_RATES)
             for g in args.graphs for n in args.n for seed in range(args.seeds)]
    runs_per_task = 1 + len(args.coverage) * (1 + len(ERROR_TYPES) * len(ERROR_RATES))
    print(f"{len(tasks)} datasets x {runs_per_task} PC runs each = {len(tasks) * runs_per_task} runs")

    start = time.time()
    graph_rows, pair_rows = [], []
    if args.jobs > 1:
        with Pool(args.jobs) as pool:
            results = pool.imap_unordered(run_one_dataset, tasks)
            for i, (g, p) in enumerate(results, 1):
                graph_rows += g
                pair_rows += p
                print(f"  {i}/{len(tasks)} datasets done ({time.time() - start:.0f}s)", flush=True)
    else:
        for i, task in enumerate(tasks, 1):
            g, p = run_one_dataset(task)
            graph_rows += g
            pair_rows += p
            print(f"  {i}/{len(tasks)} done: {task[0]} n={task[1]} seed={task[2]} "
                  f"({time.time() - start:.0f}s)", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sort_cols = ["graph", "n", "seed", "condition", "coverage", "error_type", "e"]
    graph_df = pd.DataFrame(graph_rows).sort_values(sort_cols)
    pair_df = pd.DataFrame(pair_rows).sort_values(sort_cols + ["T", "Y"])
    graph_df.to_csv(out / "graph_metrics.csv", index=False)
    pair_df.to_csv(out / "adjustment.csv", index=False)
    print(f"Saved {len(graph_df)} runs to {out / 'graph_metrics.csv'} and "
          f"{len(pair_df)} pair rows to {out / 'adjustment.csv'} in {time.time() - start:.0f}s")



if __name__ == "__main__":
    main()