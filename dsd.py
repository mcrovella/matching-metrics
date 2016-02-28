import sys
import numpy as np
import random
import networkx as nx

# randomly reconnected a disconnected graph
# will result in a star type graph for a graph with many components
def make_connected(G):
    S = nx.connected_component_subgraphs(G)
    # this is to deal with the fact that the previous function changed in 1.9.1 
    # to return a generator instead of a list
    try:
        l = len(S)
    except TypeError:
        S = list(S)
        l = len(S)
    for i in xrange(1,l):
        G.add_edge(random.sample(S[0].nodes(),1)[0],random.sample(S[i].nodes(),1)[0])

# canonical is a list of nodes in canonical order
# matrix M is input in the order of G.nodes() 
# and needs to be converted to canonical order
def reorder(M,G,canonical):
    R = np.array(M)
    n = np.shape(M)[0]
    nodeMap = {val : index for index, val in enumerate(G.nodes())}
    for i in xrange(n):
        for j in xrange(n):
            R[i][j] = M[nodeMap[canonical[i]],nodeMap[canonical[j]]]
    return R

#########################################################################
# constructing the base matrices for each kind of distance measure
#########################################################################

# assumes paths = nx.shortest_path(G)
def nhmatrix(G,paths,nodes):

    n = G.number_of_nodes()
    R = np.zeros((n,n),dtype=int)
    nodeMap = {val : index for index, val in enumerate(nodes)}
    # for pre-2.7:
    # nodeMap = dict((val,index) for (index, val) in enumerate(nodes))
    for i in xrange(n):
        nd = nodes[i]
        for j in xrange(n):
            if (i == j):
                R[i][j] = i
            else:
                R[i][j] = nodeMap[paths[nodes[j]][nd][1]]
    return R

# assumes paths = nx.shortest_path(G)
def spmatrix(G,paths,nodes):

    n = G.number_of_nodes()
    R = np.zeros((n,n),dtype=int)
    for i in xrange(1,n):
        nd = nodes[i]
        for j in xrange(i+1,n):
            R[i][j] = np.size(paths[nd][nodes[j]]) - 1
            R[j][i] = R[i][j]
    return R

# hematrix code freely borrowed from MFC's DSDmain, calcDSD, etc
def hematrix(adj):
    # adj is binary adjacency matrix

    # number of nodes
    n = np.size(adj[0])

    # p is the transition matrix of the markov chain
    p = np.zeros((n, n))

    # compute the degree of each node
    degree = np.zeros((n, 1))
    for j in xrange(n):
        degree[j] = sum(adj[j])
        # compute the transition matrix of the markov chain
        if degree[j] != 0:
            p[j] = adj[j]/degree[j]

    # from lemma 3: want (I - P + W)^-1
    c = np.eye(n)
    c = c - p
    # steady state of an undirected random walk is proportional to node degree
    pi = (degree.T)/sum(degree)
    c = c + np.tile(pi, (n, 1))
    c = np.linalg.inv(c)
    return c

# return a matrix e such that e*e' = the pseudoinverse of the laplacian of G
def ematrix(G,nodes):
    L = np.array(nx.laplacian_matrix(G,nodes))
    # not using eig.  gives very strange results for non-invertible matrices
    # which don't agree with matlab's eig.  svd however gives sensible results
    u,s,vt = np.linalg.svd(L)
    s = 1/s
    s[np.where(np.abs(s)>1e9)]=0
    s = np.sqrt(s)
    return np.dot(u,np.diag(s))

def directED(G,nodes):
    # compute effective resistance directly. 
    # when doing an entire graph, this is faster than the ELD approach
    L = nx.laplacian_matrix(G,nodes)
    Lplus = np.linalg.pinv(L)
    E = np.zeros(np.shape(L))
    n = np.shape(L)[0]
    for i in xrange(n):
        for j in xrange(n):
            E[i,j] = Lplus[i,i]+Lplus[j,j]-Lplus[i,j]-Lplus[j,i]
    return E


#########################################################################
# constructing the all-pairs distance measures
#########################################################################

def allRowNorms(mat1, mat2, normFn, LMset1, LMset2):
    # does the bookkeeping of constructing norms of every row in mat1
    # against every row in mat2, with optional restriction to a subset of
    # rows (nodeset) and a subset of columns (LMset)
    # mat1 and mat2: rows are nodes, columns are landmarks
    # normFn takes some kind of norm of every row in x against the 
    #  corresponding row in y


    # reduce He matrices to only landmark columns
    h1 = mat1[:,LMset1]
    h2 = mat2[:,LMset2]
    

    h1_nnodes = h1.shape[0]
    h2_nnodes = h2.shape[0]
    res      = np.zeros((h1_nnodes, h2_nnodes))
    # we fill out the result column-wise (i.e. we complete a single column
    # before moving on to the next. The output matrix is a 
    # len(larger) x len(smaller)-dimensional matrix. 
    #for i in xrange(0, len(smaller)):
    #    res[:,i] = normFn(larger, np.tile(smaller[i],(numNodes,1)))

    for i in xrange(0, h1_nnodes):
        for j in xrange(0, h2_nnodes):
            res[i][j] = np.sum(np.abs(h1[i] - h2[j]))

    return res

def allRowNorms_same_size(mat1, mat2, normFn, LMset_1, LMset_2):
    # does the bookkeeping of constructing norms of every row in mat1
    # against every row in mat2, with optional restriction to a subset of
    # rows (nodeset) and a subset of columns (LMset)
    # mat1 and mat2: rows are nodes, columns are landmarks
    # normFn takes some kind of norm of every row in x against the 
    #  corresponding row in y

    

    n = np.shape(mat1)[0]

    nNodes = n

    h1 = mat1[:,LMset_1]
    h2 = mat2[:,LMset_2]

    # each matrix must be over same node set and same landmark set
    assert(np.shape(h1) == np.shape(h2))

    # construct all-pairs distances according to whatever norm function is given
    res = np.zeros((nNodes,nNodes))

    # for i in xrange(nNodes):
    # # normFn: some kind of norm of every row in x against the corresponding row in y
    #     res[:,i] = normFn(h1, np.tile(h2[i],(nNodes,1)))

    for i in xrange(0, nNodes):
        for j in xrange(0, nNodes):
            res[i][j] = np.sum(np.abs(h1[i] - h2[j]))

    return res




def crossDSD(hemat1, hemat2, LMset1, LMset2):
    # return the DSD of each node in 1 to each node in 2
    # the norm function for DSD is the l1 norm of the differences

    return allRowNorms(hemat1, hemat2,
                       lambda x,y: np.sum(np.abs(x - y),1),
                       LMset1, LMset2)

def crossDSD_innerprod(D1, D2):
    n = np.shape(D1)[1]
    # set rows of hematrices to unit L2 nor
    D1n = D1 / np.tile(np.sqrt(np.sum(D1*D1,axis=1)),(n,1)).T
    D2n = D2 / np.tile(np.sqrt(np.sum(D2*D2,axis=1)),(n,1)).T
    D = D1n.dot(D2n.T)
    # for inner product, D is a similarity matrix, need to convert
    C = np.amax(D) + 1
    D = C - D
    return D


def DSD(hemat, LMset=-1, nodeset=-1):
    # hemat is matrix of He() vectors
    # return the DSD only wrt to the given set of indices

    return crossDSD(hemat, hemat, LMset, nodeset)

def crossRSD(nhmat1, nhmat2, LMset=-1, nodeset=-1):
    # return the RSD of each node in 1 to each node in 2
    # assumes the node IDs can be compared using ==

    return allRowNorms(nhmat1, nhmat2,
                       lambda x,y: np.sum(x != y,1),
                       LMset, nodeset)

def RSD(nhmat, LMset=-1, nodeset=-1):
    # nhmat is matrix of nexthop vectors
    # note that sources are on COLUMNS and destinations are on ROWS
    # so nh[i,j] is nexthop from j in the direction of i

    # return the RSD only wrt to the given set of indices

    return crossRSD(nhmat, nhmat, LMset, nodeset)

def crossLSD(spmat1, spmat2, LMset=-1, nodeset=-1):
    # return the LipSchitz distance of each node in 1 to each node in 2

    return allRowNorms(spmat1, spmat2,
                       lambda x,y: np.sqrt(np.sum((x-y)**2,1)),
                       LMset, nodeset)

def LSD(spmat, LMset=-1, nodeset=-1):
    # spmat is matrix of shortest-path lengths (or other distances)
    # so sp[i,j] is shortest path distance from i to j

    # return the LD only wrt to the given set of indices

    return crossLSD(spmat, spmat, LMset, nodeset)

def crossELD(emat1, emat2, LMset=-1, nodeset=-1):
    # return something like an effective resistance across graphs

    return allRowNorms(emat1, emat2,
                       lambda x,y: np.sum((x-y)**2,1),
                       LMset, nodeset)

def ELD(emat, LMset=-1, nodeset=-1):
    # emat is square root of pseudoinverse of lapacian
    # return the effective resistance (only wrt to the given set of landmarks)

    return crossELD(emat, emat, LMset, nodeset)

def crossESD(ermat1, ermat2, LMset=-1, nodeset=-1):
    # return the ESD of each node in 1 to each node in 2
    # the norm function for ESD is the l1 norm of the differences

    return allRowNorms(ermat1, ermat2,
                       lambda x,y: np.sum(np.abs(x - y),1),
                       LMset, nodeset)

def ESD(ermat, LMset=-1, nodeset=-1):
    # ermat is matrix of effective resistances

    return crossESD(ermat, ermat, LMset, nodeset)


#########################################################################
# code for perturbing graphs
#########################################################################

# delete each edge with probability p
def thin(G,p):
    R = nx.Graph(G)
    removeSet = [i for i in nx.edges_iter(R) if random.random() < p]
    R.remove_edges_from(removeSet)
    return R

# each edge from a to b is rewired to connect a with random node, 
# with probability p
def rewire(M,p):
    R = nx.Graph(M)
    rewireSet = [i for i in nx.edges_iter(R) if random.random() < p]
    R.remove_edges_from(rewireSet)
    for i in rewireSet:
        R.add_edge(i[0],random.sample([k for k in nx.non_neighbors(R,i[0])],1)[0])
    return R

# each edge from a to b is replaced with an edge connecting two random nodes
# with probability p
def scramble(M,p):
    R = nx.Graph(M)
    removeSet = [i for i in nx.edges_iter(R) if random.random() < p]
    R.remove_edges_from(removeSet)
    nodes = R.nodes()
    addFromNodes = [random.sample(nodes,1)[0] for i in removeSet]
    for node in addFromNodes:
        R.add_edge(node,random.sample([k for k in nx.non_neighbors(R,node)],1)[0])
    return R


# each edge or potential edge disappears or appears with probability p
# i.e., mutuate toward a G(n,p) graph
def randomize(M,p):
    R = nx.Graph(M)
    nodes = R.nodes()
    nn = np.size(nodes)
    for i in xrange(nn):
        for j in xrange(i+1,nn):
            if (random.random() < p):
                if nodes[j] in R[nodes[i]]:
                    R.remove_edge(nodes[i],nodes[j])
                else:
                    R.add_edge(nodes[i],nodes[j])
    return R

# add fraction p random edges
def addedges(M,p):
    R = nx.Graph(M)
    nodes = R.nodes()
    nn = np.size(nodes)
    for i in xrange(nn):
        for j in xrange(i+1,nn):
            if (random.random() < p) and (nodes[j] not in R[nodes[i]]):
                R.add_edge(nodes[i],nodes[j])
    return R
