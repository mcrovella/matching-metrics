from os.path import basename
import sys
import time
import argparse
import networkx as nx

_ppis = {}

def loadPPI(fname):
    G = nx.Graph()

    global _ppis

    if _ppis[fname]:
        return _ppis[fname]

    for line in open(fname, 'r'):
        p1, p2 = line.rstrip().split()
        # nx will not add duplicate edges, neither (A,B) or (B,A)
        G.add_edge(p1, p2)

    G = sorted(nx.connected_component_subgraphs(G), key = len, reverse=True)[0]

    _ppis[fname] = G

    return G
