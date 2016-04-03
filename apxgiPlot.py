import matplotlib as mp
import matplotlib.pyplot as plt
import itertools
import numpy as np
import argparse

def CIPlot(sample, ECvals, pval, n, p, gtype, ptype='none', parg=0.0):
    s = np.array(sample)
    ss = np.sort(s, axis=1)
    ns = ss.shape[1]
    low = int(pval*ns)
    hi = int((1-pval)*ns)
    med = int(ns/2)

    plt.figure()
    plt.errorbar(ECvals, ss[:,med], fmt='none', yerr=[ss[:,med]-ss[:,low],ss[:,hi]-ss[:,med]])
    plt.xlabel('Edge Correctness')
    plt.ylabel('Node Correctness')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.title('{:2.0f}% Confidence Intervals for NC as a function of EC ({}, {}, {})'.format(100*(1-2*pval),gtype,n,p))
    if (ptype != 'none'):
        plt.suptitle(r'Perturbed graph: {}, {}'.format(ptype, parg))
        plt.savefig('NCCI-n{}-p{}-{}-{}.pdf'.format(n,p,ptype,parg))
    else:
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

def rejectVals(sample, ECvals, pval):
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
        # want smallest ECval such that all ECvals[j] with (P[i] > P[j]) < pval
        try:
            # upper[i] = np.min(ECvals[(F[i,:] < pval) & (ECvals > ECvals[i])])
            upper[i] = np.min(ECvals[(ECvals > np.max(ECvals[F[i,:] > pval])) & (ECvals > ECvals[i])])
        except ValueError:
            upper[i] = np.nan
        try:
            lower[i] = np.max(ECvals[(ECvals < np.min(ECvals[F[:,i] > pval])) & (ECvals < ECvals[i])])
        except ValueError:
            lower[i] = np.nan
    return upper, lower

def rejectProfile(upper, lower, ECvals):
    import scipy.interpolate
    import scipy.signal
    uv = np.abs(upper-ECvals)
    lv = np.abs(lower-ECvals)
    uv[np.isnan(uv)] = 0.0
    lv[np.isnan(lv)] = 0.0
    single = np.max(np.array([uv,lv]),axis=0)
    f = scipy.interpolate.interp1d(ECvals, single)
    testx = np.linspace(0.05,0.95,500)
    testy = f(testx)
    sigma = 25
    window = scipy.signal.gaussian(100,sigma)
    smoothed = scipy.signal.convolve(testy, window/window.sum(), mode='valid')
    return testx[2*sigma:-(sigma-1)], smoothed

def allRejPlot(flist,pval):
    profiles = {}
    for f in flist:
        v = np.load(f)
        upper, lower = rejectVals(v['sample'],v['ECvals'],pval)
        testx, profile = rejectProfile(upper, lower, v['ECvals'])
        profiles[v['stype']] = profile

def rejectPlot(sample, ECvals, pval, n, p, gtype, ptype='none', parg=0.0):

    upper, lower = rejectVals(sample, ECvals, pval)
    plt.figure()
    plt.plot(ECvals, upper-ECvals, 'go')
    plt.plot(ECvals, lower-ECvals, 'ro')
    plt.xlabel('EC')
    plt.xlim([0,1])
    plt.ylabel(r'$\Delta$EC')
    plt.title(r'$\Delta$EC needed to reject $H_0$ at p = {}, G=({},{},{})'.format(pval,gtype,n,p))
    if (ptype != 'none'):
        plt.suptitle(r'Perturbed graph: {}, {}'.format(ptype, parg))
        plt.savefig('RejEC-n{}-p{}-pval{}-{}-{}.pdf'.format(n,p,pval,ptype,parg))
    else:
        plt.savefig('RejEC-n{}-p{}-pval{}.pdf'.format(n,p,pval))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('runfile')
    parser.add_argument('pval', type=float, default=0.005)
    args = parser.parse_args()

    v = np.load(args.runfile)

    # these aren't that useful
    # signifPlot(v['sample'], v['ECvals'], v['n'], v['p'])
    if 'ptype' in v:
        CIPlot(v['sample'], v['ECvals'], args.pval, v['n'], v['p'], v['gtype'], v['ptype'], v['parg'])
        rejectPlot(v['sample'], v['ECvals'], args.pval, v['n'], v['p'], v['gtype'], v['ptype'], v['parg'])
    else:
        CIPlot(v['sample'], v['ECvals'], args.pval, v['n'], v['p'], v['gtype'])
        rejectPlot(v['sample'], v['ECvals'], args.pval, v['n'], v['p'], v['gtype'])

