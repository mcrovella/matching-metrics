import matplotlib as mp
import matplotlib.pyplot as plt
import itertools
import numpy as np

def CIPlot(ECvals, sample, n, p):
    s = np.array(sample)
    ss = np.sort(s, axis=0)
    ns = ss.shape[1]
    low = int(0.005*ns)
    hi = int(0.995*ns)
    med = int(ns/2)

    plt.figure()
    plt.errorbar(ECvals, ss[:,med], fmt='none', yerr=[ss[:,med]-ss[:,low],ss[:,hi]-ss[:,med]])
    plt.xlabel('Edge Correctness')
    plt.ylabel('Node Correctness')
    plt.title('99% Confidence Intervals for NC as a function of EC, n={}, p={}'.format(n,p))
    plt.savefig('NCCI-n{}-p{}.pdf'.format(n,p))

# fraction of pairs (a, b) where a > b
def fracGreater(a, b):
    n = len(a)
    m = len(b)
    ng = 0
    sa = np.sort(a)[-1::-1]
    sb = np.sort(b)[-1::-1]
    ai = 0
    bi = 0
    while (ai < n and bi < m):
        if (sa[ai] > sb[bi]):
            ng += m-bi
            ai += 1
        else:
            bi += 1
    return ng/(n*m)

def signifPlot(sample, ECvals, n, p):
    s = np.array(sample)
    # build a digitization of ECvals so that we can do a sensible heatmap
    # goal: constant sized bins, only bins without values are at beginning or end
    # translates to finding a, b such that
    # s = np.sort(np.unique(np.digitize(ECvals, np.linspace(a, 1, b))))
    # s[0] = 1
    # s[-1] = len(s)
    # actually this is a good problem to analyze (later)
    # optimal value is between 1 and 0.5 times the largest intervalue gap
    bins = np.linspace(0.01, 1, 50)
    n = len(bins)
    binIndex = np.digitize(ECvals, bins)
    hm = np.zeros((n,n))
    cnt = np.zeros((n,n))
    # put each pair of ECvals in a bin
    # do fracGreater and average into bin
    for i,j in itertools.product(range(len(ECvals)), repeat=2):
        f = fracGreater(s[i,:],s[j,:])
        k = binIndex[i]
        l = binIndex[j]
        hm[k,l] += f
        cnt[k,l] += 1
    # plot heatmap
    fr = hm/cnt
    frs = fr[1:-1,1:-1]

    plt.figure()
    plt.imshow(frs.T, origin='lower')
    plt.savefig('NCSignif-n{}-p{}.pdf'.format(n,p))

def rejectPlot(sample, ECvals, pval, n, p):
    # find the upper and lower values at which one may reject the null
    # hypothesis at a given p value
    # this is equal to ECval at which the fraction of pairs for which a<b is equal to pval
    m = len(ECvals)
    ECvals = np.array(ECvals)
    s = np.array(sample)
    F = np.zeros((m,m))
    for i,j in itertools.product(range(m), repeat=2):
        F[i,j] = fracGreater(s[i,:],s[j,:])
    lower = np.zeros(m)
    upper = np.zeros(m)
    for i in range(m):
        # smallest ECval bigger than me at which (P[i] > P[j]) < pval
        try:
            upper[i] = np.min(ECvals[(F[i,:] < pval) & (ECvals > ECvals[i])])
        except ValueError:
            upper[i] = 1.0
        try:
            lower[i] = np.max(ECvals[(F[:,i] < pval) & (ECvals < ECvals[i])])
        except ValueError:
            lower[i] = 0.0

    plt.figure()
    plt.plot(ECvals, upper-ECvals, 'go')
    plt.plot(ECvals, lower-ECvals, 'ro')
    plt.xlabel('EC')
    plt.ylabel(r'$\Delta$EC')
    plt.title(r'Difference in EC necessary to reject $H_0$ at p = {}, G=({},{})'.format(pval,n,p))
    plt.savefig('RejEC-n{}-p{}-pval{}.pdf'.format(n,p,pval))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('n', type=int)
    parser.add_argument('p', type=float)
    args = parser.parse_args()

    CIPlot(sample, args.n, args.p)
    signifPlot(sample, ECvals, args.n, args.p)
    rejectPlot(sample, ECvals, 0.005, args.n, args.p)

