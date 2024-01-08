
module C1BNets

# Neural Network to do Formal Logic
#
# using a model inspired by a single Cortical Column in brain

export C1BNet, createC1BNet


using StructArrays
using ..Neuronets


struct C1BNet
  cells::StructArray{Neuron}
  axons::StructVector{Axon}
  axon_ends::Vector{Int}

  potentialConst::Ref{Float64}

  clear::Function # clear its mind 

end


function createC1BNet(
  ncols::Int=1_0000, # total number of Cortical Columns
  nmccs::Int=100, # number of cells per Mini Column
  gamma::Float64=2.2,
  min_links::Int=8, max_links::Int=10000,
  axoncap1::Int=32,
)
  @assert 1000 <= ncols <= 5_0000
  @assert 50 <= nmccs <= 5000
  @assert 2.0 <= gamma <= 3.0
  @assert 1 <= min_links <= axoncap1 < max_links

  cells = StructArray{Neuron}((
    LinearIndices((nmccs, ncols)), # uid
    zeros(Int, (nmccs, ncols)), # axonn
    zeros(Int8, (nmccs, ncols)), # potential
  ))
  ncells = length(cells)

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
  axon_ends = cumsum([cnt for (cap, cnt) in axon_capcnts])

  # number of tracked axons
  naxons = axon_ends[end]
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

  return C1BNet(
    cells, axons, axon_ends,
    potentialConst,
    clear,
  )
end


end # module C1BNets 
