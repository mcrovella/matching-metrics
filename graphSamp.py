import numpy as np
import networkx as nx
import itertools

# main reference for this code
# Benefits of Bias: Towards Better Characterization of Network Sampling
# A. Maiya and T. Y. Berger-Wolf.
# Proc. of the 17th ACM SIGKDD (KDD '11), San Diego, CA, August 2011.

def testGraph():
    return nx.Graph([[0,1],[1,2],[1,5],[2,3],[2,4],[2,5],[3,4],[4,5]])

def _BoundarySample(nodeSelect, G, n):
    current = np.random.choice(G.nodes())
    subset = {current}
    boundary = set(G.neighbors(current))
    while len(subset) < n:
        current = nodeSelect(G, subset, boundary)
        subset = subset.union({current})
        boundary = boundary.union(set(G.neighbors(current)))
        boundary -= subset
    return G.subgraph(subset)

def RandomWalkSample(G, n):
    ''' start at a random node, add nodes visited on a random walk '''
    current = np.random.choice(G.nodes())
    subset = {current}
    while len(subset) < n:
        current = np.random.choice(G.neighbors(current))
        subset = subset.union({current})
    return G.subgraph(subset)

def DegreeSample(G, n):
    ''' start at a random node, repeatedly add the neighbor with the highest degree '''
    def selectMaxDegree(G, subset, boundary):
        blist = list(boundary)
        degs = [G.degree(i) for i in blist]
        return blist[np.argmax(degs)]
    return _BoundarySample(selectMaxDegree, G, n)

def EdgeCountSample(G, n):
    ''' start at a random node, repeatedly add the neighbor with the most connections
        to the sample so far '''
    def selectMaxInducedDegree(G, subset, boundary):
        blist = list(boundary)
        induced_degs = [G.subgraph(subset.union({i})).degree(i) for i in blist]
        return blist[np.argmax(induced_degs)]
    return _BoundarySample(selectMaxInducedDegree, G, n)

def ExpansionSample(G, n):
    ''' start at a random node, repeatedly add the neighbor that expands the boundary
        by the greatest amount '''
    def selectMaxExpanding(G, subset, boundary):
        blist = list(boundary)
        allnodes = subset.union(boundary)
        new_edges = [len(set(G.neighbors(i)) - allnodes) for i in blist]
        return blist[np.argmax(new_edges)]
    return _BoundarySample(selectMaxExpanding, G, n)
        
def BFSSample(G, n):
    ''' start at a random node, add neighbor of earliest visited node '''
    current = np.random.choice(G.nodes())
    subset = {current}
    print(current)
    blist = G.neighbors(current)
    while len(subset) < n:
        current = blist.pop(0)
        print(current)
        blist += list(set(G.neighbors(current)) - subset)
        subset = subset.union({current})
    return G.subgraph(subset)

