import numpy as np
import networkx as nx
import itertools
import processPPIs
import graphSamp

def ppiGraph(n, name):

    fname = 'graphs/{}.ppi'.format(name)

    G = processPPIs.loadPPI(fname)

    # results of the sampRun experiments show that XS is best
    # at preserving EC-NC relationships
    return graphSamp.ExpansionSample(G, n)
    
