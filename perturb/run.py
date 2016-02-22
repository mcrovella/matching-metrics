from __future__ import print_function
import numpy as np
import networkx as nx
import itertools
import sys
import argparse
import dsd

sys.path.insert(0,'..')
import apxgi

def createGraph(gtype, n, p):
    if (gtype == 'ER'):
        return nx.erdos_renyi_graph(n, p)
    elif (gtype == 'BA'):
        return nx.barabasi_albert_graph(n, int(p * n))
    elif (gtype == 'WS'):
        return nx.connected_watts_strogatz_graph(n, 4, p)
    else:
        raise ValueError('Invalid graph type')

if __name__ == '__main__':

    if sys.version_info[0] < 3 or sys.version_info[1] < 5:
        print("Requires Python 3.5 or greater.")
        sys.exit(1)

    perturbFns = {'thin': dsd.thin, 'rewire': dsd.rewire, 'randomize': dsd.randomize, 'scramble': dsd.scramble}

    graphTypes = ['ER', 'BA', 'WS']

    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int, default=200)
    parser.add_argument('p', type=float, default=0.03)
    parser.add_argument('gtype', choices=graphTypes)
    parser.add_argument('ptype', choices=perturbFns.keys())
    parser.add_argument('parg', type=float)
    args = parser.parse_args()

    perturb = perturbFns[args.ptype]
    
    sample = []
    ECvals = []
    NCvals = []

    for i, nc in zip(range(200), np.linspace(0.005, 1, 200)):
        print('{}'.format(i))

        # create a random graph
        # ensure we are working with the same nodeset for both graphs
        G = createGraph(args.gtype, args.n, args.p)
        G.remove_edges_from(G.selfloop_edges())
        nodeList = G.nodes()
        while (len(list(nx.connected_components(G))) > 1):
            print('Skipping a disconnected graph.')
            G = createGraph(args.gtype, args.n, args.p)
            G.remove_edges_from(G.selfloop_edges())
            nodeList = G.nodes()
        A = np.array(nx.adj_matrix(G,nodeList).todense())

        # perturb the graph
        Gperturb = perturb(G, args.parg)
        while (len(list(nx.connected_components(Gperturb))) > 1):
            Gperturb = perturb(G, args.parg)
        B = np.array(nx.adj_matrix(Gperturb,nodeList).todense())

        try:
            correctness, EC, iters, nCands, nRejects = apxgi.ECMCMC(A, B, nc)
            # use the last n log n values as our sample
            sample.append(correctness[-iters:])
            ECvals.append(EC)
            NCvals.append(nc)
            print('rejects: {}\n****'.format(nRejects))
            if ((i % 10) == 0):
                np.savez('{}/raw/Raw-n{}-p{}-nc{}-{}-{}'.format(args.gtype,args.n,args.p,nc,args.ptype,args.parg),  correctness=correctness, EC=EC, nc=nc, n=args.n, p=args.p, gtype=args.gtype,ptype=args.ptype,parg=args.parg)
        except ValueError as err:
            print(err.args)

    sample = np.array(sample)
    ECvals = np.array(ECvals)
    np.savez('{}/Run-n{}-p{}-{}-{}'.format(args.gtype,args.n,args.p,args.ptype,args.parg), sample=sample, ECvals=ECvals, n=args.n, p=args.p, gtype=args.gtype, ptype=args.ptype, parg=args.parg)
