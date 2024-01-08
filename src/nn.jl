
module Neuronets

# Neural Network

export Neuron, Dendrite, Axon, Synapse


using StructArrays


struct Neuron
  uid::Int # unique identity
  axonn::Int # axon number
  potential::Float32 # spiking probability
end

struct Dendrite
  postn::Int # postsynaptic neuron's id
  effi::Float32 # efficacy
end

struct Axon
  pren::Int # presynaptic neuron's id 
  ndendrite::Int # number of valid dendrites
  dendrites::StructVector{Dendrite} # dendrites
end

struct Synapse
  pren::Int # presynaptic neuron's id
  postn::Int # postsynaptic neuron's id
  effi::Int8 # efficacy
end



end # module Neuronets
