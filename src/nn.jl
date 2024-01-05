
module Neuronets

# Neural Network

export Neuron, Dendrite, Axon, Synapse, Neuronet


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


struct Neuronet
  cells::StructVector{Neuron}
  axons::StructVector{Axon}

  potentialConst::Ref{Float64}

  clear::Function # clear its mind 

end


function createNeuronet(
  ncells::Int=100_0000,
  gamma::Float64=2.2,
  min_links::Int=8, max_links::Int=10000,
  axoncap1::Int=32,
)
  @assert ncells >= 100
  @assert 2.0 <= gamma <= 3.0
  @assert 1 <= min_links <= axoncap1 < ncells

  cells = StructVector{Neuron}((
    1:ncells, # uid
    zeros(Int, ncells), # axonn
    zeros(Int8, ncells), # potential
  ))

  potentialConst = Ref{Float64}(0)

  axon_capcnts = let capcnts = Tuple{Int,Int}[],
    cap = axoncap1, k = min_links, p = 0.0,
    C = ncells / sum(k^(-gamma) for k in min_links:max_links)

    while k <= max_links
      if k > cap
        push!(capcnts, (cap, Int(ceil(C * p))))
        p = k^(-gamma)
        cap *= 2
      else
        p += k^(-gamma)
      end
      k += 1
    end
    push!(capcnts, (cap, Int(ceil(C * p))))

    capcnts
  end

  # number of tracked axons
  naxons = sum(cnt for (cap, cnt) in axon_capcnts)
  @assert naxons >= ncells
  # total number of tracked synapses
  nlinks = sum(cap * cnt for (cap, cnt) in axon_capcnts)

  # storage for tracked synapses
  total_dendrites = StructVector{Dendrite}((
    zeros(Int, nlinks), # postn
    zeros(Float32, nlinks), # effi
  ))

  axons = let dendrite_begin = 1
    axon_dendrites = StructVector{Dendrite}[]
    sizehint!(axon_dendrites, naxons)
    for (cap, cnt) in axon_capcnts
      for _ in 1:cnt
        dendrite_end = dendrite_begin + cap - 1
        push!(axon_dendrites, StructVector{Dendrite}((
          total_dendrites.postn[dendrite_begin:dendrite_end],
          total_dendrites.effi[dendrite_begin:dendrite_end],
        )))
        dendrite_begin = dendrite_end + 1
      end
    end
    @assert dendrite_begin == nlinks + 1
    StructVector{Axon}((
      zeros(Int, naxons), # pren
      zeros(Int, naxons), # ndendrite
      axon_dendrites,     # dendrites
    ))
  end

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
