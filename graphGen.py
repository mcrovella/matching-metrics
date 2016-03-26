import sys
import numpy as np
import random
import networkx as nx
import itertools

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

def StickyGraph(n, deg):
    """input: n, degree sequence."""
    assert(n == len(deg))
    deg = np.array(deg) / np.sqrt(np.sum(deg))
    A = np.zeros((n,n),dtype=int)
    for i,j in itertools.combinations(range(n),2):
        if (i != j) and (np.random.random() < deg[i]*deg[j]):
            A[i,j] = 1
            A[j,i] = 1
    return nx.to_networkx_graph(A)

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

def EVGraph(n, p, q, m, ep):
    """ Create a graph according to Vazquez et al.
    VazquezGraph(n, p, q): n is number of nodes,
    p is link creation probability, q is divergence
    rate.   Start from an ER graph of size m with
    edge probability ep."""

    # Start with m nodes in ER graph, then repeat
    # n-m times:
    # Duplication: A node i is selected at random.
    # A new node i', with a link to all the
    # neighbors of i, is created.With probability p
    # a link between i' and i is established.
    # Divergence: For each of the nodes j
    # linked to i' and i we choose randomly one
    # of the two links (i', j) or (i, j) and remove it
    # with probability q.  (From Vazquez)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    # ER base graph
    for i, j in itertools.combinations(range(m),2):
        if np.random.random() < ep:
            G.add_edge(i, j)
    # DD remaining graph
    for iPrime in range(m,n):
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

def SoleGraph(n, delta, beta):
    """ Create a graph according to Sole et al.
    SoleGraph(n, delta, beta). Start from a ring of 5 nodes.
    Each time step:
    (a) one node in the graph is randomly chosen and
    duplicated; (b) the links emerging from the new generated
    node are removed with probability delta; (c) finally,
    new links (not previously present) can be created between the new
    node and all the rest of the nodes with probability alpha.
    Note that we modify this so that if a node duplication results
    in a disconnected graph, we consider this as a failed duplication
    and remove the node."""
    alpha = beta/n
    G = nx.Graph()
    G.add_nodes_from(range(n))
    # base ring graph
    for i in range(5):
        G.add_edge(i, (i+1)%5)
    # iterate
    for iPrime in range(5, n):
        connected = False
        while (not connected):
            i = np.random.random_integers(0,iPrime-1,1)[0]
            # duplicate
            for j in G.neighbors(i):
                if j != iPrime:
                    G.add_edge(j, iPrime)
            # remove edges
            for j in G.neighbors(iPrime):
                if np.random.random() < delta and j != i:
                    G.remove_edge(j, iPrime)
            # add edges
            for j in range(iPrime):
                if np.random.random() < alpha and not G.has_edge(j, iPrime):
                    G.add_edge(j, iPrime)
            connected = G.degree(iPrime) > 0
    return G

