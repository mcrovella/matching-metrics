import sys
import numpy as np
import random
import networkx as nx

def geoGraph(n, d, epsilon):
    """ Create a geometric graph: n points in d-dimensional space,
    nodes are connected if closer than epsilon"""
    points = np.random.random((n,d))
    pl2 = np.array([np.linalg.norm(points, axis=1)])**2
    eucDist = (pl2.T @ np.ones((1,n))) + (np.ones((n,1)) @ pl2) - (2 * points @ points.T)
    A = ((eucDist + np.eye(n)) < epsilon).astype(int)
    return nx.to_networkx_graph(A)



def geoGraphP(n, d, p):
    """ Create a geometric graph: n points in d-dimensional space,
    fraction p node pairs are connected"""
    points = np.random.random((n,d))
    pl2 = np.array([np.linalg.norm(points, axis=1)])**2
    eucDist = (pl2.T @ np.ones((1,n))) + (np.ones((n,1)) @ pl2) - (2 * points @ points.T)
    dists = np.sort(np.ravel(eucDist))
    epsilon = dists[n + np.floor((n**2-n) * p).astype(int)]
    A = ((eucDist + dists[-1] * np.eye(n)) < epsilon).astype(int)
    return nx.to_networkx_graph(A)

# def StickyGraph(
#     """input: n, degree sequence.  degrees are assigned to nodes.  normalize sum of all
#    products to sum to 1

def VazquezGraph(n, p, q):
    """ Create a graph according to Vazquez et al.
    VazquezGraph(n, p, q): n is number of nodes,
    p is link creation probability, q is divergence
    rate."""

    # Start with 2 connected nodes, then repeat
    # n-2 times:
    # Duplication: A node i is selected at random.
    # A new node i', with a link to all the
    # neighbors of i, is created.With probability p
    # a link between i' and i is established.
    # Divergence: For each of the nodes j
    # linked to i' and i we choose randomly one
    # of the two links (i', j) or (i, j) and remove it
    # with probability q.  (From Vazquez)

    A = np.zeros((n,n),dtype='int')
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edge(0,1)
    for iPrime in range(2,n):
        i = np.random.random_integers(0,iPrime-1,1)[0]
        if np.random.random() < p:
            G.add_edge(iPrime, i) 
        for j in G.neighbors(i):
            if j != iPrime:
                G.add_edge(j, iPrime)
        for j in G.neighbors(iPrime):
            if np.random.random() < q and j != i:
                G.remove_edge(j, np.random.permutation([iPrime, i])[0])
    return G
