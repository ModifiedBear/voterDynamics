using GLMakie
using ThreadsX
using CircularArrays
using LinearAlgebra
using StatsBase
using Random
using OffsetArrays

function hamiltonian(selected_spin::CartesianIndex{2}, space::CircularArray, interaction::CircularArray, locs::Vector{CartesianIndex{2}})
# TODO: create type for locs for the function to not allocate anything
# OR: make them "CartesianIndices"

neighbour_spin = view(space,fill(selected_spin,length(locs)) + locs)
neighbour_interaction = view(interaction,fill(selected_spin,length(locs)) + locs)
H = (-dot(neighbour_interaction, neighbour_spin) * view(space, selected_spin))[1] # can be either 0,±2, or ±4

# μ = 0.1
# H -= 
# H += 

# calculate energy of neighbours

return H
end

function update_space(MAXITER, space::CircularArray, indices::CartesianIndices, interaction::CircularArray, locs::Vector{CartesianIndex{2}},β::Float64)
_space = copy(space)
magnetization = zeros(MAXITER)
selected_spin = CartesianIndex{2}

for ITER in 1:MAXITER
    # selected_spin = 
  neighbours    = fill(rand(indices),length(locs)) + locs
  selected_spin = rand(indices)
  proposed_space = copy(_space)
  proposed_space[selected_spin] *= -1
  energy_old = sum(hamiltonian.( [selected_spin;neighbours], Ref(_space), Ref(interaction), Ref(locs)))
  energy_new = sum(hamiltonian.( [selected_spin;neighbours], Ref(proposed_space), Ref(interaction), Ref(locs)))

  if energy_new < energy_old
    _space = proposed_space
  elseif exp(-β*(energy_new - energy_old)) > rand()
    _space= proposed_space
  else
    nothing
  end

  magnetization[ITER] = mean(_space)
end
return _space, magnetization
end

using CairoMakie
begin
GC.gc()

NRUNS       = 20
N           = 40
rng         = MersenneTwister(253)
space       = CircularArray(sample(rng,[-1,1], Weights([0.59, 0.41]),(N,N)) )
interaction = CircularArray(ones(N,N))
indices     = eachindex(space)
locs        = CartesianIndex.([-1,0,0,1],[0,1,-1,0])
BOLTZMANN   = 1.0
TEMP        = 1.0
BETA        = 1/(BOLTZMANN*TEMP)
MAXITER     = 50000
β = BETA
# function recursive_ising(S::CircularArray, )
#   if iter == 1
#     return 
#   else
#     S = recursive_ising()
# M = zeros(MAXITER, NRUNS)
# _space = copy(space)
# magnetization = zeros(MAXITER)

S,_ = update_space(MAXITER, space, indices, interaction,locs, BETA)
@allocated M = ThreadsX.map((i)->last(update_space(MAXITER, space, indices, interaction,locs, BETA)), 1:NRUNS)
end

let 
  fig = Figure()
  ax = [Axis(fig[1,1]), Axis(fig[1,2]), Axis(fig[2,1:2])]
  heatmap!(ax[1], space.data, colormap=:bone)
  heatmap!(ax[2], S.data, colormap=:bone)
  lines!.(ax[3], M, color=(Makie.wong_colors()[2], 0.4))

  lines!(ax[3], mean(M), color=:black)
  fig
end