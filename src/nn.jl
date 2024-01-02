
module Neuronets

# Neural Network

export Neuron, Neuronet


using StructArrays


const Timestep = Int64


struct Neuron
  uid::Int # unique identity
  potential::Int8 # spiking probability
  axonn::Int # axon number
end

struct Dentrite
  postn::Int # postsynaptic neuron's id
  effi::Int8 # efficacy
end

struct Axon
  pren::Int # presynaptic neuron's id 
  dendrites::StructVector{Denrite} # dendrites
  ndendrite::Int # number of valid dendrites
end

struct Synapse
  pren::Int # presynaptic neuron's id
  postn::Int # postsynaptic neuron's id
  effi::Int8 # efficacy
end


struct Neuronet
  cells::StructVector{Neuron}
  axons::StructVector{Axon}

  function Neuronet(
    ncells::Int=100_0000, nlinks::Int=30_0000_0000;
    gamma::Float64=2.2,
  )
    @assert ncells >= 100
    @assert nlinks >= 2 * ncells
    @assert 2.0 <= gamma <= 3.0

    return new(
      StructVector{Neuron}((
        1:ncells, # uid
        zeros(Int8, ncells), # potential
        zeros(Int, ncells), # axonn
      )),
      StructVector{Axon}((
        zeros(Int, nlinks), # pren
      ))
    )

  end
end


function _axons(
  ncells::Int=100_0000, nlinks::Int=30_0000_0000;
  gamma::Float64=2.2, min_links::Int=8, axoncap1::Int=32,
)
  @assert ncells >= 100
  @assert nlinks >= 2 * ncells
  @assert 2.0 <= gamma <= 3.0
  @assert 2 <= min_links <= axoncap1 < ncells


end



end # module Neuronets
