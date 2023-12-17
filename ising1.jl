# ising-like interactions on a (non periodic) graph
# TODO: create type for weights ij to be iterable or smnthn

using Graphs, SimpleWeightedGraphs, StatsBase
using Distributions
using LinearAlgebra
using SparseArrays
using Printf
using Random
using CairoMakie, GraphMakie
using GLMakie

graph_size   = 1000
connectivity = 10
# rng = Random.MersenneTwister(433)

destinations = sample(rng,1:graph_size, connectivity*graph_size,replace=true)
sources = shuffle(rng, destinations)
influence = rand(rng, Uniform(0.0,1.0),length(sources)); # can get weights greater than 1.0, but it can be normalized (for now), it is a random graph anyway
popul_graph = SimpleWeightedGraph(sources, destinations, influence)

# remove self influences
if has_self_loops(popul_graph)
  vertices_w_loops = Iterators.flatten(simplecycles_limited_length(popul_graph,1))
  rem_edge!.(Ref(popul_graph), vertices_w_loops, vertices_w_loops)
end

# connect hermits
no_neighbours = isempty.(neighbors.(Ref(g), vertices(g)))
has_neighbours = findall(.~no_neighbours)
add_edge!.(Ref(g), findall(no_neighbours), sample(rng,has_neighbours, sum(no_neighbours)))

# normalize weights
# g.weights = g.weights/maximum(g.weights) # normalize weights (works for random graph, shouldn't be required for real data)

unlooped_influence = findnz(triu(Graphs.weights(g)))[3]
unlooped_influence .= 0.5  # full influence

SPIN_COLORS = Dict(tuple.((:down, :up),cgrad(:coolwarm,2, categorical=true)))
spin_vector = sample(rng, [-1,1],StatsBase.Weights([0.1,0.9]), nv(g))

#= plot stuff
# weights_label    = map((w)->@sprintf("%.4f",w), unlooped_influence)
COL_RANGE        = (minimum(unlooped_influence),maximum(unlooped_influence))
NODE_COL         = :white # theme_dark().attributes[:backgroundcolor].val;
CMAP             = :batlow;
NODE_LABELS      = string.(1:nv(g))
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

function get_influence(g::SimpleWeightedGraph, inode::Int)
  # equivalent to w_ij of Ising model
  return view(g.weights,inode,neighbors(g,inode))
end

function hamiltonian(h::SimpleWeightedGraph, spin::Vector)
  # full hamiltonian, used for testing only
  return -dot.(get_influence.(Ref(h), vertices(h)), view.(Ref(spin), neighbors.(Ref(h),vertices(h))) .* spin[vertices(h)])
end

function efficient_hamiltonian(h::SimpleWeightedGraph, spin::Vector, idx::Int)
  # calculate hamiltonian for σ at idx
  return -dot(get_influence(g, idx), view(spin, neighbors(g,idx)))*spin[idx]
end


function update_spin(β::Float64,  graph::SimpleWeightedGraph,orig_spin::Vector)
  spin_vec = copy(orig_spin)
  node_to_flip = rand(rng, 1:nv(graph))

  # β  = 0.1
  # old_energy_full = sum(hamiltonian(graph, spin_vector))
  old_energy = sum(efficient_hamiltonian.(Ref(graph), Ref(spin_vec), [neighbors(graph, node_to_flip); node_to_flip]))

  spin_vec[node_to_flip] *= -1

  # new_energy_full = sum(hamiltonian(graph, spin_vec))
  new_energy = sum(efficient_hamiltonian.(Ref(graph), Ref(spin_vec), [neighbors(graph, node_to_flip); node_to_flip]))

  if (new_energy - old_energy) < 0 # accept new configuration
    return spin_vec
  elseif exp(-β*(new_energy - old_energy)) >= 1 # accept new configuration
    return spin_vec
  else # reject changes
    return orig_spin
  end
end

# begin process
begin
GC.gc()
BOLTZMANN = 1.0
TEMP      = 10.0
BETA      = 1/(BOLTZMANN*TEMP)
MAXITER   = 1050
NUM_RUNS  = 50

states    = fill(spin_vector, NUM_RUNS) # must be vector of vectors

people_up = zeros(Int64, MAXITER, NUM_RUNS)
people_down = zeros(Int64, MAXITER, NUM_RUNS)

isGreater(s::Vector) = s .> 0
isSmaller(s::Vector) = s .< 0

people_up[1,:] = permutedims(sum.(map(isGreater, states)))
people_down[1,:] = permutedims(sum.(map(isSmaller, states))) # 1 - above

# STATESUM = sum(abs.(diff(states)))

# while (STATESUM > 0 && ITER < MAXITER)
for (time,(step_up, step_down)) in enumerate(zip(eachrow(people_up),eachrow(people_down)))
  @printf("time = %i\n", time)
  states = update_spin.(Ref(g),states)
  step_up[:] = sum.(map(isGreater, states))
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
# lines!.(ax, eachcol(people_down), linestyle=:solid,color=(SPIN_COLORS[:down], 0.1))
lines!.(ax, eachcol(people_up), linestyle=:solid,color=(SPIN_COLORS[:up], 0.1))
# lines!(ax, vec(mean(people_down, dims=2)), color=SPIN_COLORS[:down], label=L"\left\langle\sum\downarrow\right\rangle")
lines!(ax, vec(mean(people_up, dims=2)), color=SPIN_COLORS[:up], label=L"\left\langle\sum\uparrow\right\rangle")
# axislegend(ax, unique=true, orientation=:horizontal, position=:ct)
fig
end