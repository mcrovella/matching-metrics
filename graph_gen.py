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
    return A


