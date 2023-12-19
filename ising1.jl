# ising-like interactions on a (non periodic) graph
# TODO: create type for weights ij to be iterable or smnthn

using Graphs, SimpleWeightedGraphs, StatsBase, NetworkLayout
using Distributions
using LinearAlgebra
using SparseArrays
using Printf
using Random
using CairoMakie, GraphMakie
using GLMakie

begin
# rng = Random.MersenneTwister(433)
graph_size   = 100
connectivity = 4
destinations = sample(1:graph_size, connectivity*graph_size,replace=true)
sources = shuffle(destinations)
influence = rand(Uniform(0.0,1.0),length(sources)); # can get weights greater than 1.0, but it can be normalized (for now), it is a random graph anyway
# popul_graph = SimpleWeightedGraph(sources, destinations, influence)
nx = 20
popul_graph = SimpleWeightedGraph(grid([nx,nx]))

# x connect
[add_edge!(popul_graph,i ,(nx^2-(nx-i)), 1.0) for i in 1:nx]
# y connect
[add_edge!(popul_graph, i , i+nx-1, 1.0) for i in 1:nx:nx^2]

# popul_graph = SimpleGraph(100, 100, StochasticBlockModel([2,2],1, [1000,1]))
# popul_graph = smallgraph(:tutte)
# repeat for second population

# modify graph a little
# second_graph_size = 50
# add_vertices!(popul_graph, second_graph_size)
# new_vertices = graph_size+1:(graph_size+second_graph_size)
# new_influence = rand(Uniform(0.0, 1.0), length(new_vertices))
# [add_edge!.(Ref(popul_graph), new_vertices, shuffle(new_vertices), new_influence) for _ in 1:connectivity]

# remove self influences
# if has_self_loops(popul_graph)
#   vertices_w_loops = Iterators.flatten(simplecycles_limited_length(popul_graph,1))
#   rem_edge!.(Ref(popul_graph), vertices_w_loops, vertices_w_loops)
# end

# # connect both clouds
# # add_edge!(popul_graph, sample(1:graph_size), sample(new_vertices), 0.5)

# connect hermits
# no_neighbours = isempty.(neighbors.(Ref(popul_graph), vertices(popul_graph)))
# has_neighbours = findall(.~no_neighbours)
# add_edge!.(Ref(popul_graph), findall(no_neighbours), sample(has_neighbours, sum(no_neighbours)))

# # normalize weights
# # g.weights = g.weights/maximum(g.weights) # normalize weights (works for random graph, shouldn't be required for real data)

# unlooped_influence = findnz(triu(Graphs.weights(popul_graph)))[3]
# unlooped_influence .= 0.5  # 50/50 influence

spin_vector = sample([-1,1],StatsBase.Weights([0.5,0.5]), nv(popul_graph))
SPIN_COLORS = Dict(tuple.((:down, :up),cgrad(:coolwarm,2, categorical=true)))
NODE_LABELS      = string.(1:nv(popul_graph))
# visualize graph
fig = Figure()
ax = Axis(fig[1,1])
graphplot!(ax, popul_graph, node_color=spin_vector, node_attr=(colorrange=(-1,1),colormap=colormap=[SPIN_COLORS[:down], SPIN_COLORS[:up]]), nlabels=NODE_LABELS)
fig
end

include("./src/Ising.jl")

#= plot stuff
# weights_label    = map((w)->@sprintf("%.4f",w), unlooped_influence)
COL_RANGE        = (minimum(unlooped_influence),maximum(unlooped_influence))
NODE_COL         = :white # theme_dark().attributes[:backgroundcolor].val;
CMAP             = :batlow;
NODE_LABELS      = string.(1:nv(popul_graph))
SPIN_LABELS      = string.(spin_vector)
AX_SETTINGS = (xtickalign=1, ytickalign=1, xgridvisible=false, ygridvisible=false, backgroundcolor=:transparent,bottomspinevisible=true,topspinevisible=true, leftspinevisible=true, rightspinevisible=true)

spin_dn = [MarkerElement(color = SPIN_COLORS[1], marker=:rect, markersize = 15)]
spin_up = [MarkerElement(color = SPIN_COLORS[2], marker=:rect, markersize = 15)]

fig = Makie.Figure()
ax = Axis(fig[1,1]; AX_SETTINGS...)
GraphMakie.graphplot!(ax, g, node_color=NODE_COL,nlabels=SPIN_LABELS, edge_color=unlooped_influence, elabels=weights_label,nlabels_color=spin_vector, nlabels_attr=(colorrange=(-1,1), colormap=SPIN_COLORS),node_strokewidth=0.0,node_size=20,ilabels=NODE_LABELS, edge_attr=(colorrange=COL_RANGE,colormap=CMAP))
# # GraphMakie.graphplot!(ax, g, node_color=spin_vector,  edge_color=unlooped_influence,nlabels_color=:red,node_strokewidth=0.0,node_size=10, node_attr=(colorrange=(-1,1), colormap=[:blue, :red]), edge_attr=(colorrange=COL_RANGE,colormap=CMAP))
axislegend(ax,   [spin_up, spin_dn], ["+1", "-1"])
Colorbar(fig[1,2], label="Prob", colormap=CMAP, limits = COL_RANGE)
fig
=# 


# begin process
begin
GC.gc()
BOLTZMANN = 1.0
TEMP      = 10.0e2
BETA      = 1/(BOLTZMANN*TEMP)
MAXITER   = 350
NUM_RUNS  = 1

states    = fill(spin_vector, NUM_RUNS) # must be vector of vectors
# people_up = zeros(Int64, MAXITER, NUM_RUNS)
# people_down = zeros(Int64, MAXITER, NUM_RUNS)

# isGreater(s::Vector) = s .> 0
# isSmaller(s::Vector) = s .< 0

magnetized = zeros(Float64, MAXITER, NUM_RUNS)

# people_up[1,:] = permutedims(sum.(map(isGreater, states)))
# people_down[1,:] = permutedims(sum.(map(isSmaller, states))) # 1 - above

# STATESUM = sum(abs.(diff(states)))

# while (STATESUM > 0 && ITER < MAXITER)
for (time,magnet) in enumerate(eachrow(magnetized))
  @printf("time = %i\n", time)
  states = Ising.update_spin.(BETA, Ref(popul_graph),states)
  magnet[:]  = Ising.magnetization.(states)
  # step_down[:] = sum.(map(isSmaller, states)) # 1 - above
  # graphplot!(ax, g, node_color=states,edge_color=unlooped_influence,node_size=20, node_attr=(colorrange=(-1,1), colormap=SPIN_COLORS), edge_attr=(colorrange=COL_RANGE, colormap=CMAP))
  # save("./images/fig"*lpad(ITER, 5, '0')*".png", fig)
  # STATESUM = sum(abs.(diff(states)))
  # ITER += 1
end
end

let

GC.gc()
fig = Figure()
ax = Axis(fig[1,1], xlabel="iter", ylabel="n")
lines!.(ax, eachcol(magnetized), color=(Makie.wong_colors()[2], 0.5))
lines!(ax, vec(mean(magnetized, dims=2)), color=:black, label=L"\left\langle\sum\uparrow\right\rangle")
axislegend(ax, unique=true, orientation=:horizontal, position=:ct)
ylims!(ax, [-1.05,1.05])
fig
end

let
display_graph = SimpleWeightedGraph(grid([nx,nx]))

edgecolors = fill((:black, 0.2),1:ne(display_graph))
fig2 = Figure()
ax2 = Axis(fig2[1,1])
graphplot!(ax2, display_graph, node_color=states[1],edge_color=edgecolors, node_attr=(colorrange=(-1,1), colormap=[SPIN_COLORS[:down], SPIN_COLORS[:up]]), layout=NetworkLayout.Spring(dim=2, C=0.1), node_size=15)
fig2
end
