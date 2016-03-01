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

function deltaMat(A, B, P)
    # this was the slow step - n^3
    # P = A @ B
    n = A.shape[0]
    K = np.ones((n,1),dtype=int) @ [np.diag(P)]
    T = K + K.T - (P + P.T + 2 * A * B)
    return T
