from graphs import nonadjacent_pairs
import random
import json
import sys 

def correct_knowledge(dag, coverage):
    knowledge={'required':[], 'forbidden':[]}
    edges = list(dag['edges'])
    forbidden_edges = list(nonadjacent_pairs(dag))
    n_correct = round(coverage * len(edges))

    required = random.sample(edges, n_correct)

    knowledge['required'] = required

    forbidden = random.sample(forbidden_edges, len(forbidden_edges))
    knowledge['forbidden'] = forbidden
    print(knowledge)

data = json.loads(sys.stdin.read())

correct_knowledge(data,0.75)
# def error_knowledge(K, dag, coverage):
                         