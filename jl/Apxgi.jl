module Apxgi

# This contains a julia traslation of apxgi.py

function swap!(l, i, j)
         tmp = l[i]
         l[i] = l[j]
         l[j] = tmp
end

function swapRowsCols!(B, i, j)
         tmp = B[i,:]
         B[i,:] = B[j,:]
         B[j,:] = tmp
         tmp = B[:,i]
         B[:,i] = B[:,j]
         B[:,j] = tmp
end

function updateP!(P, A, B, i, j)
    diff = B[i,:] - B[j,:]
    diff[i] = 0
    diff[j] = 0
    # D12 is 2 x n
    D12 = [-diff; diff]
    # A11,A21 @ D12 -> n x n
    Pup = [A[:,i] A[:,j]] * D12
    # A12,A22 @ D21 -> n X 2
    Pupcols = A * D12'
    Pup[:,i] = Pupcols[:,1] - Pup[:,i]
    Pup[:,j] = Pupcols[:,2] - Pup[:,j]
    P += Pup
end

# should replace the second line with a broadcast
function deltaMat(A, B, P)
    # this was the slow step - n^3
    # P = A @ B
    n = size(A)[1]
    K = ones(Int, (n,1)) * reshape(diag(P),(1,n))
    K + K' - (P + P' + 2 * (A & B))
end

function ECMCMC(A, B, startingNC, nIters)
      correctness = []
      nCands = []

      n = size(A)[1]
      iters = round(Int, n * ceil(log(n)/log(2)))

      # start with a permutation that has some number of correct node mappings
      # as a way to influence the edge correctness setting for this run
      nCorrect = round(Int,ceil(n * startingNC))
      # important to choose the correctly-matched nodes randomly
      perm = zeros(Int, n)
      pmap = randperm(n)
      correctidx = pmap[1:nCorrect]
      incorrectidx = pmap[nCorrect+1:end]
      perm[correctidx] = correctidx
      perm[incorrectidx[randperm(n-nCorrect)]] = incorrectidx
      oldPerm = copy(perm)

      # create a permutation matrix
      Pi = eye(Int,n)[perm,:]'

      # permute the node mappings
      B = Pi' * B * Pi
      oldB = copy(B)

      # P is A'B
      # T is the test matrix such that if T(i,j) == 0, then i and j can be swapped (i != j)
      P = A * B
      T = deltaMat(A, B, P)
      nOverlaps = trace(P)
      nOldOverlaps = nOverlaps
      oldT = copy(T)
      oldP = copy(P)

      # determine the set of legal transitions
      # the size of the set is the degree of this state in the Markov chain
      candidates = findn((T + eye(n)).==0)
      m = length(candidates[1])
      if (m == 0)
         error(@sprintf("Mapping has no neighbors for nc=%f", startingNC))
      end
      oldM = m
      # printf("ncandidates = %d',m)

      EC = nOverlaps/trace(A' * A)
      NC = sum(perm .== (1:n))
      @printf("NC: %0.5f.  Edges matching: %d, EC: %0.5f\n",NC/n,nOverlaps,EC)

      nRejects = 0
    
      for i in 1:(nIters*iters)

        # determine the set of legal transitions
        # the size of the set is the degree of this state in the Markov chain
        candidates = findn((T + eye(n)).==0)
        m = length(candidates[1])
        if (m == 0)
            error(@sprintf("Mapping has no neighbors for nc=%f", startingNC))
        end
        # printf("ncandidates = %d',m)

        # safety check that we never change the edge correctness of our mapping
        assert(nOverlaps == nOldOverlaps)

        # decide whether to accept this new state according to Metropolis dynamics
        # we accept transition to this new state with probability min(1, olddegree/newdegree)
        # conceptually we are biasing the walk away from higher degree nodes
        # in fact we are guaranteeing that the steady state of the chain is the uniform dist
        if (oldM < m) & (rand() > oldM/m)
            # reject transition 
            nRejects += 1
            perm = oldPerm
            m = oldM
            T = oldT
            B = oldB
            P = oldP
            candidates = findn((T + eye(n)).==0)
        end

        # save this state so it can be reverted if needed
        oldPerm = copy(perm)
        oldM = m
        oldT = copy(T)
        oldB = copy(B)
        oldP = copy(P)

        # compute the number of correct node mappings
        push!(correctness,sum(perm .== (1:n)))
        push!(nCands, m)
        # @printf("Node correctness: %d\n",sum((1:n) .== perm))

        # choose a transition at random
        c = rand(1:m)
        i, j = candidates[1][c], candidates[2][c]
        ## print('swapping {} and {}'.format(candidates[0][c],candidates[1][c]))
        ## print('old perm: {}'.format(perm))
    
        # keep track of the permutation in case we want to print it
        swap!(perm, i, j)
        ## print('new perm: {}'.format(perm))

        # we could create a new permutation matrix, recompute B, etc
        # but that would be slow so we just permute the current node mappings

        # apply to P the effect of permuting B 
        # P = A'B, but recomputing it is too slow
        updateP!(P, A, B, i, j)

        # permute B
        swapRowsCols!(B, i, j)

        # compute new T
        T = deltaMat(A, B, P)

        # test the number of overlaps
        nOldOverlaps = nOverlaps
        nOverlaps = trace(P)

    end
      
    correctness = correctness / n

return correctness, EC, iters, nCands, nRejects



end

# module Apxgi
end 
