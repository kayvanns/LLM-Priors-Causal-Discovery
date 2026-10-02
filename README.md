# When Can Causal Discovery Trust Its Priors?
### Error propagation from LLM-supplied background knowledge to causal conclusions in hybrid constraint-based discovery

**Research question.** How do different types of error in LLM-supplied background knowledge propagate through the PC algorithm to downstream causal conclusions? Which types of knowledge can safely be imposed as constraints?

**Core quantity.** For each error type, the probability that the discovered graph still yields a valid confounder adjustment set, as a function of the background-knowledge error rate *e*:

$$
P(\text{valid adjustment set} \mid e,\ \text{error type},\ n)
$$

The **tipping point** *e\** is the error rate at which the benefit of background knowledge disappears: the smallest *e* at which P(valid adjustment set) is no longer significantly higher than what PC achieves with no background knowledge.

---

## Motivation

Constraint-based causal discovery methods such as PC ([Spirtes et al., 2000](#references)) can recover a causal graph from observational data only up to its Markov equivalence class, and they also make finite-sample testing errors. Domain knowledge is the usual remedy: an expert states that "A cannot cause B" or "A causes B," and the algorithm incorporates it ([Meek, 1995](#references)). Large language models are now proposed as a scalable source of that knowledge:

- **Kıcıman, Ness, Sharma & Tan (TMLR 2024)** show that LLMs answer pairwise causal-direction questions with high accuracy. They argue that LLMs should be *combined with* data-driven causal methods rather than replace them, while noting that LLMs fail in unpredictable ways.
- **Vashishtha et al. (2023)** use LLM-elicited causal orderings to identify adjustment sets for effect estimation, linking LLM knowledge directly to downstream causal conclusions.
- **Srivastava et al. (NeurIPS 2025)** show that standard benchmarks (BNLearn graphs such as ALARM and Asia) are memorized by LLMs. On new expert-consensus graphs published after LLM training cutoffs, LLM-only discovery performs far worse, but a simple LLM + PC hybrid outperforms both LLM-only and purely statistical methods. They leave open whether negative priors ("no edge here") help, concluding that "noisy or incorrect priors may in fact impair overall performance. More work is needed."

Hybrid pipelines treat LLM output as knowledge, yet that output is noisy in ways that are hard to predict. Prior work evaluates hybrids at whatever error rate a given LLM happens to have on a given benchmark. **This project controls the error directly.** It injects specific types of error into otherwise-correct background knowledge at known rates, then measures how each type changes the learned graph and the causal conclusions drawn from it. Because the error is synthetic, the results do not depend on any particular LLM and cannot be inflated by benchmark memorization.

---

## Benchmarks

Both graphs come from Srivastava et al. (2025). Each is an expert-consensus causal graph from a recent scientific study, rather than a classic benchmark likely to be memorized.

| Graph | Nodes | Edges | Confounded (T → Y) pairs | Source |
|---|---|---|---|---|
| Alzheimer's disease | 11 | 19 | 13 | Consensus of 5 domain experts (edges kept if ≥ 2 agreed), 2023 study |
| COVID-19 respiratory | 11 | 20 | 26 | Iterative elicitation with 7–12 experts, 2022 study |

A *confounded pair* is a treatment → outcome pair for which the true DAG requires adjustment, meaning the empty set is not a valid backdoor set. Examples include `tau → moca` (tau pathology on cognition, confounded by age and amyloid) and `viremia → systemic inflammatory response`. Pairs that need no adjustment are excluded, because any learned graph passes them trivially.

Edge lists are in `graphs/*.json`. The Alzheimer's graph was transcribed from the published figure; the directions of `tau → ventricular_volume` and `brain_volume → ventricular_volume` should be checked against the source study.

---

## How knowledge enters PC

Every claim in the knowledge base is enforced as a hard constraint, regardless of what the data says:

- **Required edge `X → Y`:** the X–Y adjacency is never removed in the skeleton phase, even if a conditional independence test finds X ⫫ Y | S. It is then oriented as claimed before Meek's rules orient the remaining edges. This matches the LLM + PC design of Srivastava et al.
- **Forbidden edge `X — Y`:** the adjacency is removed unconditionally.

causal-learn removes forbidden edges natively, but its required edges only orient adjacencies that survive the independence tests. `scripts/discover.py` therefore wraps the conditional independence test so that every required pair always tests as dependent, which guarantees the edge is kept. With no constraints, the wrapper reproduces causal-learn's `pc` output exactly.

Enforcing every claim means each injected error actually reaches the learned graph, so the experiment measures the full effect of acting on wrong knowledge.

---

## Methodology

### 1. Data generation
For each graph, data are simulated from a linear Gaussian structural equation model, matching Srivastava et al.:

$$
X_j = \sum_{i \in \mathrm{Pa}(j)} w_{ij} X_i + \varepsilon_j, \qquad w_{ij} \sim U(0, 2),\quad \varepsilon_j \sim \mathcal{N}(0, 1)
$$

- Sample sizes: *n* ∈ {250, 1,000, 5,000}
- Seeds: 20 per condition. Each seed draws new weights; the same seed uses the same weights at every *n*, so sample sizes differ only in the amount of data.
- PC uses a Fisher-z test with α = 0.05 (causal-learn).
- Baseline: PC with no background knowledge, for every graph, *n* and seed.

### 2. Correct knowledge base
For each seed, a correct knowledge base K\* is built from the true DAG:
- **Required claims:** a random 75% of the true edges, each with its true direction (`X → Y`)
- **Forbidden claims:** an equal number of truly non-adjacent pairs

75% coverage is the main condition, and 50% is a comparison. Coverage is set high because PC alone yields a valid adjustment set for fewer than 5% of confounded pairs on these graphs. With correct knowledge, P(valid) rises to about 0.25 at 50% coverage and 0.5–0.6 at 75%, which leaves room to measure how much of that gain errors take away.

### 3. Controlled error injection
Starting from K\*, a fraction *e* ∈ {0, 0.1, 0.2, 0.3, 0.4, 0.5} of the relevant claims is corrupted with **one error type at a time**:

| Error type | What the informant says | Construction |
|---|---|---|
| **False positive** | `X → Y` exists, but X and Y are not adjacent | Replace a fraction *e* of required claims with required claims on random non-adjacent pairs that are not already forbidden |
| **False negative** | X and Y are not adjacent, but `X → Y` exists | Replace a fraction *e* of forbidden claims with forbidden claims on true edges. These are drawn first from true edges not in the required set; if more are needed, they are drawn from required edges, which are then removed from the required set. |
| **Wrong direction** | `X → Y` when the truth is `Y → X` | Reverse a fraction *e* of required claims |

A false negative is what results when an omission is treated as a prohibition, as when negative priors are built from edges an LLM did not predict (Srivastava et al., 2025).

### 4. Outcomes

**Graph-level** (learned graph against the true DAG):
- Structural Hamming distance (SHD): pairs that are missing, extra, reversed or left undirected
- Adjacency precision, recall and F1
- Arrowhead (orientation) precision, recall and F1

**Decision-level (primary):** for each of the 39 confounded pairs (T, Y):
1. Read the adjustment set off the learned graph: Z = parents of T. If T has any undirected or bidirected edge, the learned graph does not identify a set; this is recorded as *not identified* and counted as a failure. If the learned graph makes Y a parent of T, it is recorded as *reversed* and counted as a failure.
2. Check whether Z is a valid backdoor adjustment set for (T, Y) in the **true** DAG.
3. **Materially different conclusion:** estimate the effect of T on Y by OLS of Y on T and Z. As the reference, estimate it the same way on the same data using a correct adjustment set (the true parents of T), with its 95% confidence interval. A corrupted-background-knowledge analysis is considered materially different when its effect estimate falls outside the 95% confidence interval of the effect estimate obtained using the correct adjustment set, or when the estimated effect reverses sign. Pairs that are not identified or reversed count as materially different. The exact true effect (the sum, over all directed paths from T to Y, of the products of edge weights) is also recorded.

### 5. Analysis
*To be completed.*

### 6. Figures
*To be completed.*

---

## Directory structure

```
LLM_Causal_Discovery/
├── README.md
├── requirements.txt
├── graphs/
│   ├── alzheimers.json          # 11 nodes, 19 edges (Srivastava et al., 2025)
│   └── covid_respiratory.json   # 11 nodes, 20 edges (Srivastava et al., 2025)
└── scripts/
    ├── graphs.py      # load DAGs, confounded (T, Y) pairs, backdoor check, exact total effects
    ├── simulate.py    # linear Gaussian SEM data generation
    ├── discover.py    # PC with hard required and forbidden edge constraints
    └── metrics.py     # SHD, adjacency and arrowhead P/R/F1, adjustment validity, material difference
```

---

## References

- Kıcıman, E., Ness, R., Sharma, A., & Tan, C. (2024). Causal reasoning and large language models: Opening a new frontier for causality. *Transactions on Machine Learning Research.* [arXiv:2305.00050](https://arxiv.org/abs/2305.00050)
- Srivastava, A., Nagalapatti, L., Jajoo, G., Vashishtha, A., Krishnamurthy, P., & Sharma, A. (2025). Realizing LLMs' causal potential requires science-grounded, novel benchmarks. *NeurIPS 2025.* [arXiv:2510.16530](https://arxiv.org/abs/2510.16530)
- Vashishtha, A., Reddy, A. G., Kumar, A., Bachu, S., Balasubramanian, V. N., & Sharma, A. (2023). Causal inference using LLM-guided discovery. [arXiv:2310.15117](https://arxiv.org/abs/2310.15117)
- Meek, C. (1995). Causal inference and causal explanation with background knowledge. *Proceedings of UAI.*
- Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation, Prediction, and Search* (2nd ed.). MIT Press.
