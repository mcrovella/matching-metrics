import numpy as np
import networkx as nx
import itertools
import processPPIs

def ppiGraph(n, name):

    G = processPPIs.loadPPI(fname)
