module Ising

using Graphs, SimpleWeightedGraphs, LinearAlgebra, Random

export magnetization,update_spin

function neighbour_weights(g::SimpleWeightedGraph, inode::Int)
  # equivalent to w_ij of Ising model
  return view(g.weights,inode,neighbors(g,inode))
end

function hamiltonian(h::SimpleWeightedGraph, spin::Vector)
  # full hamiltonian of Ising lattice, used for testing only
  return -dot.(neighbour_weights.(Ref(h), vertices(h)), view.(Ref(spin), neighbors.(Ref(h),vertices(h))) .* spin[vertices(h)])
end

function efficient_hamiltonian(h::SimpleWeightedGraph, spin::Vector, idx::Int)
  # calculate hamiltonian for σ at idx
  # to use neighbors -> neighborhood, weights have to represent
  # distance, and one then can choose a geodesic radius
  return -dot(neighbour_weights(h, idx), view(spin, neighbors(h,idx)))*spin[idx]
end

function magnetization(spin::Vector)
  # calculate magnetization (number of "up" spins)
  return sum(spin) / length(spin)
end

function update_spin(β::Float64,  graph::SimpleWeightedGraph,orig_spin::Vector)
  # runs ising process
  spin_vec = copy(orig_spin)
  node_to_flip = rand(1:nv(graph))

  # β  = 0.1
  # old_energy_full = sum(hamiltonian(graph, spin_vector))
  old_energy = sum(efficient_hamiltonian.(Ref(graph), Ref(spin_vec), [neighbors(graph, node_to_flip); node_to_flip]))

  spin_vec[node_to_flip] *= -1

  # new_energy_full = sum(hamiltonian(graph, spin_vec))
  new_energy = sum(efficient_hamiltonian.(Ref(graph), Ref(spin_vec), [neighbors(graph, node_to_flip); node_to_flip]))

  if (new_energy - old_energy) < 0 # accept new configuration
    return spin_vec
  elseif exp(-β*(new_energy - old_energy)) > rand() # accept new configuration
    return spin_vec
  else # reject changes
    return orig_spin
  end
end


end # module ising
