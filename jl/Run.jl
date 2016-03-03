import Apxgi
reload("Apxgi")
using NPZ
        
function geoGraph(n, d, p)
	 points = rand((n,d))
	 pl2 = sum(points.^2,2)
     eucDist = broadcast(+,pl2,broadcast(+,pl2',-2 * points * points'))
     dists = sort(eucDist[:])
     epsilon = dists[n+1+round(Int,(n^2-n)*p,RoundDown)]
     A = round(Int, (eucDist + dists[end]*eye(n)) .< epsilon)
end

graphTypes = ["GEO"]

n = 200
p = 0.2
gtype = "GEO"

sample = []
ECvals = []
NCvals = []

steps = 3

tic()
for (i, nc) in zip(1:steps, linspace(1/steps, 1, steps-1/steps))

    @printf("%d/%d\n",i,steps)

    A = geoGraph(n, 3, p)

    # remove self edges
    # need to ensure connectedness and skip disconnected graphs
    # optionally perturb the graph

    try
        correctness, EC, iters, nCands, nRejects = Apxgi.ECMCMC(A, A, nc, 5)
        push!(sample, correctness[end-iters+1:end])
        push!(ECvals, EC)
        push!(NCvals, nc)
        @printf("rejects: %d\n****\n",nRejects)
        if ((i % 10) == 0)
           fname = @sprintf("noperturb/%s/raw/Raw-n%d-p%f-nc%f", gtype, n, p, nc)
           data = Dict{UTF8String, Any}("correctness" => correctness, "EC" => EC, "nc" => nc,
                                "n" => n, "p" => p, "gtype" => gtype)
           # npzwrite(fname, data)
        end
    catch err
       print(err,"\n")
    end
end

toc()
# fname = @sprintf("noperturb/%s/Run-n%d-p%f", gtype, n, p)
# data = Dict{UTF8String, Any}("sample" => sample, "ECvals" => ECvals, "n" => n, "p" => p)
# npzwrite(fname, data)
