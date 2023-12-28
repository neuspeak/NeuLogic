module NeuLogic


export BInt8
include("./bint8.jl")
using .BInt8Type


include("./c1bnet.jl")
using .C1BNet


end # module NeuLogic
