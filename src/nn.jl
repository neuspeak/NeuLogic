
module Neuronets

# Neural Network

export Neuron, Neuronet


using StructArrays


const Timestep = Int64


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


struct Neuronet
  cells::StructVector{Neuron}
  axons::StructVector{Axon}

  potentialConst::Ref{Float64}

  clear::Function # clear its mind 

end


function createNeuronet(
  ncells::Int=100_0000, nlinks::Int=30_0000_0000;
  gamma::Float64=2.2,
  min_links::Int=8, max_links::Int=10000,
  axoncap1::Int=32,
)
  @assert ncells >= 100
  @assert nlinks >= 2 * ncells
  @assert 2.0 <= gamma <= 3.0
  @assert 2 <= min_links <= axoncap1 < ncells

  cells = StructVector{Neuron}((
    1:ncells, # uid
    zeros(Int8, ncells), # potential
    zeros(Int, ncells), # axonn
  ))

  potentialConst = Ref{Float64}(0)


  axon_capcnts = let capcnts = Tuple{Int,Int}[],
    cap = axoncap1, k = min_links, p = 0.0,
    C = 1.0 / sum([k^(-gamma) for k in min_links:max_links])

    while k <= max_links
      if k > cap
        push!(capcnts, (cap, Int(floor(nlinks * C * p / cap))))
        p = k^(-gamma)
        cap *= 2
      else
        p += k^(-gamma)
      end
      k += 1
    end
    push!(capcnts, (cap, Int(floor(nlinks * C * p / cap))))

    @assert sum([cap * cnt for (cap, cnt) in capcnts]) <= nlinks

    capcnts
  end


  axons = StructVector{Axon}((
    zeros(Int, nlinks), # pren
    zeros(Int, nlinks), # ndendrite
    StructVector{Dendrite}((
      zeros(Int, nlinks), # postn
    ))
  ))

  function clear()
    cells.potential[:] = [
      r <= 1 ? 1.0 : r^(-gamma) for r in cells.axonn
    ]
    potentialConst[] = 1.0 / sum(cells.potential)
  end

  return Neuronet(
    cells, axons,
    potentialConst,
    clear,
  )
end



end # module Neuronets
