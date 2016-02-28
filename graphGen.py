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
    A = ((eucDist + np.eye(n)) < epsilon).astype(int)
    return nx.to_networkx_graph(A)

# def StickyGraph(
#     """input: n, degree sequence.  degrees are assigned to nodes.  normalize sum of all
#    products to sum to 1


