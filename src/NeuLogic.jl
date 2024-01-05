module NeuLogic


export Neuron, Neuronet
include("./nn.jl")
using .Neuronets


include("./c1bnet.jl")
using .C1BNet


end # module NeuLogic
