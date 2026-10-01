# When Can Causal Discovery Trust Its Priors?
### Error propagation from LLM-supplied background knowledge to causal conclusions in hybrid constraint-based discovery

**Research question.** How do different types of error in LLM-supplied background knowledge propagate through the PC algorithm to downstream causal conclusions? Which types of knowledge can safely be imposed as constraints?

**Core quantity.** For each error type, the probability that the discovered graph still yields a valid confounder adjustment set, as a function of the background-knowledge error rate *e*:

$$
P(\text{valid adjustment set} \mid e,\ \text{error type},\ n)
$$

The **tipping point** *e\** is the smallest error rate at which this probability falls below the no-knowledge baseline. Beyond *e\**, adding the "knowledge" makes the causal conclusions worse than ignoring it.

---

## Motivation

Constraint-based causal discovery methods such as PC ([Spirtes et al., 2000](#references)) can recover a causal graph from observational data only up to its Markov equivalence class. In practice they also suffer from finite-sample testing errors. Domain knowledge is the usual remedy: an expert says "A cannot cause B" or "A causes B," and the algorithm incorporates it ([Meek, 1995](#references)). Large language models are now proposed as a scalable source of that knowledge:

- **Kıcıman, Ness, Sharma & Tan (TMLR 2024)** show that LLMs answer pairwise causal-direction questions with high accuracy. They argue that LLMs should be *combined with*, not replace, data-driven causal methods, while noting that LLMs fail in unpredictable ways.
- **Vashishtha et al. (2023)** use LLM-elicited causal orderings to identify adjustment sets for effect estimation, linking LLM knowledge directly to downstream causal conclusions.
- **Srivastava et al. (NeurIPS 2025)** show that standard benchmarks (BNLearn graphs such as ALARM and Asia) are memorized by LLMs. They introduce new, expert-consensus graphs published after LLM training cutoffs. On these graphs, LLM-only discovery scores far lower than on the old benchmarks, but a simple LLM + PC hybrid outperforms both LLM-only and purely statistical methods. They leave open whether *negative* priors ("no edge here") help, concluding that "noisy or incorrect priors may in fact impair overall performance. More work is needed."

The gap is this: hybrid pipelines treat LLM output as knowledge, yet LLM output is noisy in ways that are hard to predict. Prior work evaluates hybrids at whatever error rate a given LLM happens to have on a given benchmark. **This project instead controls the error directly.** It injects specific types of error into otherwise-correct background knowledge at known rates, then measures how each type changes the learned graph and, more importantly, the causal conclusions drawn from it.

Because the error is synthetic, the results do not depend on any particular LLM and cannot be inflated by benchmark memorization. Any LLM's measured error profile can later be placed on these curves to predict whether its knowledge is safe to use.

---

## Benchmarks

Both graphs come from Srivastava et al. (2025). Each is an expert-consensus causal graph from a recent scientific study, rather than a classic benchmark likely to be memorized.

| Graph | Nodes | Edges | Confounded (treatment → outcome) pairs | Source |
|---|---|---|---|---|
| Alzheimer's disease | 11 | 19 | 13 | Consensus of 5 domain experts (edges kept if ≥ 2 agreed), 2023 study |
| COVID-19 respiratory | 11 | 20 | 26 | Iterative elicitation with 7–12 experts, 2022 study |

A *confounded pair* is a treatment → outcome pair for which the true DAG requires adjustment, meaning the empty set is not a valid backdoor set. Examples include `tau → moca` (tau pathology on cognition, confounded by age and amyloid) and `viremia → systemic inflammatory response`. Pairs that need no adjustment are excluded, because any learned graph passes them trivially.

Edge lists are in `data/graphs/*.json`. The Alzheimer's graph was transcribed from the published figure; the directions of `tau → ventricular_volume` and `brain_volume → ventricular_volume` should be checked against the source study.

---

## How knowledge enters PC

Every claim in the knowledge base is enforced as a hard constraint, regardless of what the data says:

- **Required edge `X → Y`:** the X–Y adjacency is never removed in the skeleton phase, even if a conditional independence test says X ⫫ Y | S. It is then oriented as claimed before Meek's rules orient the remaining edges. This matches the LLM + PC design of Srivastava et al.
- **Forbidden edge `X — Y`:** the adjacency is removed unconditionally.

Enforcing every claim means each injected error actually reaches the learned graph, so the experiment measures the full effect of acting on wrong knowledge.

---

## Methodology

### 1. Data generation
For each graph, data are simulated from a linear Gaussian structural equation model, matching Srivastava et al.:

$$
X_j = \sum_{i \in \mathrm{Pa}(j)} w_{ij} X_i + \varepsilon_j, \qquad w_{ij} \sim U(0, 2),\quad \varepsilon_j \sim \mathcal{N}(0, 1)
$$

- Sample sizes: *n* ∈ {250, 1,000, 5,000}
- Seeds: 20 per condition. Each seed draws new weights and new data.
- PC uses a Fisher-z test with α = 0.05 (causal-learn).
- Baseline: PC with no background knowledge, for every graph, *n* and seed.

### 2. Correct knowledge base
For each seed, a correct knowledge base K\* is built from the true DAG:
- **Required claims:** a random 50% of the true edges, each with its true direction (`X → Y`)
- **Forbidden claims:** an equal number of truly non-adjacent pairs (`X — Y` absent)

The 50% coverage reflects an informant who knows some, but not all, of the structure. Coverage is varied in a sensitivity analysis.

### 3. Controlled error injection
Starting from K\*, a fraction *e* ∈ {0, 0.1, 0.2, 0.3, 0.4, 0.5} of the relevant claims is corrupted with **one error type at a time**:

| Error type | What the informant says | Construction |
|---|---|---|
| **False positive** | `X → Y` exists, but X and Y are not adjacent | Replace a fraction *e* of required claims with required claims on random non-adjacent pairs |
| **False negative** | Fails to mention a real relationship | Drop a fraction *e* of required claims. In the *open-world* reading the omission adds no constraint. In the *closed-world* reading, which mirrors how Srivastava et al. build negative priors from edges an LLM did not predict, the omitted edge becomes forbidden. |
| **Wrong direction** | `X → Y` when the truth is `Y → X` | Reverse a fraction *e* of required claims |
| **False forbidden** | `X → Y` cannot exist, but it does | Replace a fraction *e* of forbidden claims with forbidden claims on true edges |

The open-world false negative is a control: it only reduces how much knowledge is available and should never perform worse than baseline.

### 4. Outcomes

**Graph-level** (learned graph against the true DAG):
- Structural Hamming distance (SHD): missing, extra or reversed edges
- Adjacency precision, recall and F1
- Arrowhead (orientation) precision, recall and F1

**Decision-level (primary):** for each of the 39 confounded pairs (T, Y):
1. Read the adjustment set off the learned graph: Z = parents of T. If T has any undirected edge, the learned graph does not identify a set; this is recorded as *not identified* and counted as a failure.
2. Check whether Z is a valid backdoor adjustment set for (T, Y) in the **true** DAG (`pgmpy.CausalInference.is_valid_backdoor_adjustment_set`).
3. **Effect-estimate bias:** in a linear SEM the true total effect of T on Y is the sum, over all directed paths from T to Y, of the products of edge weights. Estimate it by OLS of Y on T and Z, and record the relative bias and whether the sign flips. A *materially different conclusion* is defined as relative bias above 20% or a sign flip.

### 5. Analysis
- **Validity curves:** P(valid adjustment set) against *e*, one panel per error type, with the no-knowledge baseline overlaid. Fit with logistic regression, with 95% confidence intervals from a bootstrap clustered by seed.
- **Tipping point:** *e\** is the smallest error rate at which the curve falls significantly below the no-knowledge baseline, reported per error type, graph and *n*.
- **Safety table:** the final summary ranks the four knowledge types by tipping point, from most to least robust to error, and flags which are *safe at realistic LLM error rates* and which are *unsafe even at low error*.

Planned figures:
1. SHD against *e* by error type
2. P(valid adjustment set) against *e*, the main result
3. Tipping point *e\** against sample size, testing whether more data makes the pipeline more or less tolerant of bad knowledge

---

## Pilot result

This pilot used an earlier version of the design: COVID-19 graph, *n* = 2,000, 10 seeds, and 10 hints per run. Values are mean SHD; lower is better.

| Hints wrong | No hints | Forbidden | Required |
|---|---|---|---|
| 0% | 16.4 | 15.6 | 7.3 |
| 20% | 16.4 | 17.6 | 9.9 |
| 40% | 16.4 | 18.0 | 13.7 |
| 60% | 16.4 | 19.3 | 17.0 |

Required edges cut errors by more than half when correct and kept an advantage up to about 40% wrong. Forbidden edges helped little when correct and hurt as soon as any were wrong. On the BNLearn ALARM network, required edges broke even at about 20% error, compared with about 60% here. This suggests the tipping point depends on how informative the data is by itself. The full experiment tests that directly (Figure 3).

---

## Directory structure

```
llm-causal-priors/
├── README.md
├── requirements.txt
├── data/
│   └── graphs/
│       ├── alzheimers.json          # 11 nodes, 19 edges (Srivastava et al., 2025)
│       └── covid_respiratory.json   # 11 nodes, 20 edges (Srivastava et al., 2025)
├── src/
│   ├── graphs.py         # load DAGs, list confounded (T, Y) pairs, true total effects
│   ├── simulate.py       # linear Gaussian SEM data generation
│   ├── knowledge.py      # correct knowledge base K* and the four error injectors
│   ├── discover.py       # PC with required and forbidden edge constraints (causal-learn)
│   ├── metrics.py        # SHD, adjacency and arrowhead P/R/F1, adjustment validity, effect bias
│   └── run_experiments.py   # full grid: graph × n × error type × e × seed
├── results/
│   ├── graph_metrics.csv     # one row per run
│   └── adjustment.csv        # one row per run × (T, Y) pair
├── notebooks/
│   └── analysis.ipynb        # validity curves, tipping points, figures
├── figures/
└── writeup/
    └── report.pdf            # short methods and results write-up
```

## Reproducing

```bash
pip install -r requirements.txt     # causal-learn, pgmpy, numpy, pandas, networkx, statsmodels, matplotlib
python src/run_experiments.py --graphs alzheimers covid_respiratory --n 250 1000 5000 --seeds 20
jupyter nbconvert --execute notebooks/analysis.ipynb
```

---

## Limitations

- **Linear Gaussian data.** This favors PC with Fisher-z tests. Results may differ under nonlinear mechanisms.
- **Errors are random within each type.** A real LLM's errors are probably systematic; for example, it may confuse correlated symptoms more often. The curves here describe the random-error case.
- **Causal sufficiency.** PC assumes no hidden confounders. Clinical data rarely satisfies this.
- **Graph size.** Both graphs have 11 variables, and Srivastava et al. find that hybrid gains shrink on larger graphs.

---

## Connection to my research and next steps

My lab work at UW–Madison uses ensemble causal discovery (PC, FCI) on MIMIC-IV sepsis records. That pipeline surfaced a candidate effect of mechanical ventilation on acute kidney injury, which I am now testing with a target trial emulation. Both steps depend on background knowledge: clinicians tell the algorithm which relationships are impossible, and the adjustment sets for the trial emulation come from the resulting graph. This project asks what my own pipeline implicitly assumes: **how wrong can that knowledge be before the adjustment set, and the clinical conclusion built on it, is wrong?** As LLMs start to stand in for the clinician, the answer becomes a practical requirement for using them safely.

More broadly, I want to develop causal methods that make observational data trustworthy enough for consequential decisions in medicine and policy, where randomized trials are impractical or unethical. Knowing when to trust the prior knowledge going into a causal analysis is a central part of that.

**Extensions to this project:**
1. **Real LLM knowledge.** Elicit pairwise claims from models whose training cutoff predates each graph's publication, following the memorization-free principle of Srivastava et al. Measure each model's error profile by type and place it on the validity curves to predict whether its knowledge is safe to use.
2. **Latent confounding.** Repeat the experiment with FCI and adjustment criteria for PAGs, which is closer to clinical reality and to my lab pipeline.
3. **Nonlinear mechanisms.** Use the paper's MLP-based SEM with a kernel conditional-independence (KCI) test.
4. **Systematic error models.** Make errors concentrate on semantically similar or highly correlated variable pairs, as LLM errors plausibly do, instead of falling at random.
5. **Confidence filtering.** Keep only the claims an LLM rates as high-confidence, and test whether that lowers the effective error rate below the tipping point.
6. **A new clinical benchmark.** Build an expert-consensus graph for ventilation, kidney injury and sepsis with clinicians from my lab, using the same recipe as Srivastava et al. That would give a graph no model has seen, from the setting I care most about.

---

## References

- Kıcıman, E., Ness, R., Sharma, A., & Tan, C. (2024). Causal reasoning and large language models: Opening a new frontier for causality. *Transactions on Machine Learning Research.* [arXiv:2305.00050](https://arxiv.org/abs/2305.00050)
- Srivastava, A., Nagalapatti, L., Jajoo, G., Vashishtha, A., Krishnamurthy, P., & Sharma, A. (2025). Realizing LLMs' causal potential requires science-grounded, novel benchmarks. *NeurIPS 2025.* [arXiv:2510.16530](https://arxiv.org/abs/2510.16530)
- Vashishtha, A., Reddy, A. G., Kumar, A., Bachu, S., Balasubramanian, V. N., & Sharma, A. (2023). Causal inference using LLM-guided discovery. [arXiv:2310.15117](https://arxiv.org/abs/2310.15117)
- Meek, C. (1995). Causal inference and causal explanation with background knowledge. *Proceedings of UAI.*
- Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation, Prediction, and Search* (2nd ed.). MIT Press.
