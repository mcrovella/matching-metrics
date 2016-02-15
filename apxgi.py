from __future__ import print_function
import numpy as np
import networkx as nx
import itertools
import sys
import argparse
import pickle

if sys.version_info[0] < 3 or sys.version_info[1] < 5:
    print("Requires Python 3.5 or greater.")
    sys.exit(1)

def swap(l, i, j):
    tmp = l[i]
    l[i] = l[j]
    l[j] = tmp

def swapRowsCols(B, i, j):
    tmp = B[i,:].copy()
    B[i,:] = B[j,:]
    B[j,:] = tmp
    tmp = B[:,i].copy()
    B[:,i] = B[:,j]
    B[:,j] = tmp
    
# def updateDeltaMat(A, B, P, T):

# this is n^3 + BigConstant*n^2
# turns out to be slower in practice, at least for small graphs, than deltaMat
def updateDeltaMat(n, A, B, P, T):
    P = A @ B
    for i, j in itertools.product(range(n), range(n)):
        T[i,j] = P[i,i]+P[j,j]-P[i,j]-P[j,i]-2*A[i,j]*B[i,j]

# this is 3 n^3
def deltaMat(A, B):
    P = A @ B
    n = A.shape[0]
    Pdiag = np.diag(np.diag(P))
    OnesMat = np.ones((n,n),dtype=int) 
    T = Pdiag @ OnesMat + OnesMat @ Pdiag - (P + P.T + 2 * A * B)
    return P, T

def ECMCMC(A, startingNC, nIters = 5):

    correctness = []
    nCands = []

    n = A.shape[0]
    iters = n * np.ceil(np.log(n)/np.log(2)).astype('int')

    # start with a permutation that has some number of correct node mappings
    # as a way to influence the edge correctness setting for this run
    correct = np.ceil(n * startingNC).astype('int')
    incorrect = n - correct
    perm = list(range(correct))+list(correct + np.random.permutation(incorrect))
    oldPerm = perm.copy()

    # create a permutation matrix
    Pi = np.eye(n,dtype=int)[perm].T

    # permute the node mappings
    B = Pi.T @ A @ Pi

    # P is A'B
    # T is the test matrix such that if T(i,j) == 0, then i and j can be swapped (i != j)
    P, T = deltaMat(A, B)
    nOverlaps = np.trace(P)
    nOldOverlaps = nOverlaps
    oldT = T.copy()

    # determine the set of legal transitions
    # the size of the set is the degree of this state in the Markov chain
    candidates = np.where((T + np.eye(n))==0)
    m = len(candidates[0])
    if (m == 0):
        raise ValueError('Mapping has no neighbors for nc={}'.format(nc))
    oldM = m
    # print('ncandidates = {}'.format(m))

    EC = nOverlaps/np.trace(A.T @ A)
    print('{}, {}'.format(nOverlaps,EC))

    nRejects = 0
    
    for i in range(nIters*iters):

        # determine the set of legal transitions
        # the size of the set is the degree of this state in the Markov chain
        candidates = np.where((T + np.eye(n))==0)
        m = len(candidates[0])
        if (m == 0):
            raise ValueError('Mapping has no neighbors for n={}, p={}, nc={}'.format(n,p,nc))
        ## print('ncandidates = {}'.format(m))

        # safety check that we never change the edge correctness of our mapping
        assert(nOverlaps == nOldOverlaps)

        # decide whether to accept this new state according to Metropolis dynamics
        # we accept transition to this new state with probability min(1, olddegree/newdegree)
        # conceptually we are biasing the walk away from higher degree nodes
        # in fact we are guaranteeing that the steady state of the chain is the uniform dist
        if (oldM < m) and np.random.random() > oldM/m:
            # reject transition 
            nRejects += 1
            perm = oldPerm
            m = oldM
            T = oldT
            B = oldB
            candidates = np.where((T + np.eye(n))==0)

        # save this state so it can be reverted if needed
        oldPerm = perm.copy()
        oldM = m
        oldT = T.copy()
        oldB = B.copy()

        # compute the number of correct node mappings
        correctness.append(np.sum(np.array(range(n))==perm))
        nCands.append(m)
        ## print('Node correctness: {}'.format(np.sum(np.array(range(n))==perm)))

        # choose a transition at random
        c = np.random.random_integers(0,m-1,1)[0]
        i, j = candidates[0][c], candidates[1][c]
        ## print('swapping {} and {}'.format(candidates[0][c],candidates[1][c]))
        ## print('old perm: {}'.format(perm))
    
        # keep track of the permutation in case we want to print it
        swap(perm, i, j)
        ## print('new perm: {}'.format(perm))

        # we could create a new permutation matrix, recompute B, etc
        # but that would be slow so we just permute the current node mappings
        swapRowsCols(B, i, j)

        # compute P and T
        P, T = deltaMat(A, B)

        # test the number of overlaps
        nOldOverlaps = nOverlaps
        nOverlaps = np.trace(P)

    correctness = np.array(correctness) / n

    return correctness, EC, iters, nCands, nRejects

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int, default=200)
    parser.add_argument('p', type=float, default=0.03)
    args = parser.parse_args()

    sample = []
    ECvals = []
    NCvals = []

    for i, nc in zip(range(200), np.linspace(0.005, 1, 200)):
        print('{}'.format(i))

        # create a random graph
        G = nx.erdos_renyi_graph(args.n, args.p)
        while (len(list(nx.connected_components(G))) > 1):
            print('Skipping a disconnected graph.')
            G = nx.erdos_renyi_graph(args.n, args.p)
        A = np.array(nx.adj_matrix(G).todense())

        try:
            correctness, EC, iters, nCands, nRejects = ECMCMC(A, nc)
            # use the last n log n values as our sample
            sample.append(correctness[-iters:])
            ECvals.append(EC)
            NCvals.append(nc)
            print('rejects: {}\n****'.format(nRejects))
        except ValueError as err:
            print(err.args)

    sample = np.array(sample)
    ECvals = np.array(ECvals)
    np.savez('Run-n{}-p{}'.format(args.n,args.p), sample=sample, ECvals=ECvals)
