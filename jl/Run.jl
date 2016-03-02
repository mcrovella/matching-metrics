
function geoGraph(n, d, p)
	 points = rand((n,d))
	 pl2 = sum(points.^2,2)
     eucDist = broadcast(+,pl2,broadcast(+,pl2',-2 * points * points'))
     dists = sort(eucDist[:])
     epsilon = dists[n+1+round(Int,(n^2-n)*p,RoundDown)]
     A = round(Int, (eucDist + dists[end]*eye(n)) .< epsilon)
end


sample = []
ECvals = []
NCvals = []

steps = 500



