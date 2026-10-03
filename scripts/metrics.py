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