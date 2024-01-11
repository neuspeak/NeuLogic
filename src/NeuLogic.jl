module NeuLogic


export Neuron, Dendrite, Axon, Synapse
include("./nn.jl")
using .Neuronets


export C1BNet, createC1BNet
include("./c1bnet.jl")
using .C1BNets


export C1BConcept
include("./c1bc.jl")
using .C1BConcepts


end # module NeuLogic
