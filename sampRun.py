import numpy as np
import networkx as nx
import itertools
import sys
import argparse
import dsd
import apxgi
import graphGen
import time
import graphSamp

def createGraph(gtype, n, p):
    if (gtype == 'ER'):
        return nx.erdos_renyi_graph(n, p)
    elif (gtype == 'BA'):
        return nx.barabasi_albert_graph(n, int(p * n))
    elif (gtype == 'WS'):
        return nx.connected_watts_strogatz_graph(n, 8, p)
    elif (gtype == 'GEO'):
        return graphGen.geoGraphP(n, 3, p)
    elif (gtype == 'VZ'):
        # Vazquez recommends p = 0.1, q = 0.7
        # Gibson suggests p = 0.24, q = 0.887
        qmap = {0.1:0.7, 0.24:0.887}
        assert(p in qmap.keys())
        return graphGen.VazquezGraph(n, p, qmap[p])
    elif (gtype == 'EV'):
        qmap = {0.1:0.7, 0.24:0.887}
        assert(p in qmap.keys())
        return graphGen.EVGraph(n, p, qmap[p], n//5, 0.8)
    elif (gtype == 'SL'):
        # values from Sole paper
        return graphGen.SoleGraph(n, 0.53, 0.06)
    else:
        raise ValueError('Invalid graph type')

if __name__ == '__main__':

    if sys.version_info[0] < 3 or sys.version_info[1] < 5:
        print("Requires Python 3.5 or greater.")
        sys.exit(1)

    graphTypes = ['GEO', 'SL']
    sampleTypes = ['None', 'RW', 'DEG', 'EC', 'XS', 'BFS']

    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int, default=500)
    parser.add_argument('p', type=float, default=0.03)
    parser.add_argument('gtype', choices=graphTypes)
    parser.add_argument('ptype', type=str, nargs='?', default='noperturb')
    args = parser.parse_args()

    subSample = {}
    subECVals = {}
    subNVal = {}

    sampFn = {'RW': graphSamp.RandomWalkSample,
              'DEG': graphSamp.DegreeSample,
              'EC': graphSamp.EdgeCountSample,
              'XS': graphSamp.ExpansionSample,
              'BFS': graphSamp.BFSSample,
              'None': lambda G, n: G}

    steps = 10

    for sampType in sampleTypes:
        subSample[sampType] = []
        subECVals[sampType] = []

    for i, nc in zip(range(steps), np.linspace(1/(steps+1), 1, steps, endpoint=False)):
        print('{}/{}'.format(i,steps))
        # create a random graph
        # ensure we are working with the same nodeset for both graphs
        G = createGraph(args.gtype, args.n, args.p)
        G.remove_edges_from(G.selfloop_edges())
        while (len(list(nx.connected_components(G))) > 1):
            print('Skipping a disconnected graph.')
            G = createGraph(args.gtype, args.n, args.p)
            G.remove_edges_from(G.selfloop_edges())

        # apply each kind of sampling (including no sampling) to the graph
        # and compute its EC vs NC profile
        for sampType in sampleTypes:
            print('sampling type {}'.format(sampType))

            Gsamp = sampFn[sampType](G, args.n//2)
            nodeList = Gsamp.nodes()
            n = len(nodeList)
            print('nodes: {}'.format(n))
            subNVal[sampType] = n

            Asamp = np.array(nx.adj_matrix(Gsamp,nodeList).todense())
            Bsamp = Asamp.copy()

            try:
                correctness, EC, iters, nCands, nRejects = apxgi.ECMCMC(Asamp, Bsamp, nc)
                # use the last n log n values as our sample
                subSample[sampType].append(correctness[-iters:])
                subECVals[sampType].append(EC)
                print('rejects: {}\n****'.format(nRejects))
                if ((i % 10) == 0):
                    if (args.ptype == 'noperturb'):
                        np.savez('samprun/noperturb/{}/raw/Raw-n{}-p{}-nc{}-s{}'.format(args.gtype,n,args.p,nc,sampType),  correctness=correctness, EC=EC, nc=nc, n=n, p=args.p, gtype=args.gtype, stype=sampType)
                    else:
                        np.savez('samprun/perturb/{}/raw/Raw-n{}-p{}-nc{}-{}-{}-s{}'.format(args.gtype,n,args.p,nc,args.ptype,args.parg,sampType),  correctness=correctness, EC=EC, nc=nc, n=n, p=args.p, gtype=args.gtype,ptype=args.ptype,parg=args.parg,stype=sampType)
            except ValueError as err:
                print(err.args)


    # finally, output all collected data
    for sampType in sampleTypes:
        sample = np.array(subSample[sampType])
        ECvals = np.array(subECVals[sampType])
        nval = subNVal[sampType]
        if (args.ptype == 'noperturb'):
            np.savez('samprun/noperturb/{}/Run-n{}-p{}-s{}'.format(args.gtype,nval,args.p,sampType), sample=sample, ECvals=ECvals, n=nval, p=args.p, gtype=args.gtype, stype=sampType)
        else:
            np.savez('samprun/perturb/{}/Run-n{}-p{}-{}-{}-s{}'.format(args.gtype,nval,args.p,args.ptype,args.parg,sampType), sample=sample, ECvals=ECvals, n=nval, p=args.p, gtype=args.gtype, ptype=args.ptype, parg=args.parg, stype=sampType)

    t = time.process_time()
    print('Ended: n={}, p={}, {}, {}, steps={}, elapsed time={:.2f} secs ({:.2f} secs/step).'.format(args.n,args.p,args.gtype,args.ptype,steps,t,t/steps))
