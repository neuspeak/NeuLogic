module NeuLogic


export Neuron, Dendrite, Axon, Synapse
include("./nn.jl")
using .Neuronets


export C1BNet, createC1BNet
include("./c1bnet.jl")
using .C1BNets


end # module NeuLogic
