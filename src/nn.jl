
module Neuronets

# Neural Network

export Neuron, Dendrite, Axon


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
  nsynapse::Int # number of valid synapses
  synapses::StructVector{Dendrite} # synapses
end



end # module Neuronets
