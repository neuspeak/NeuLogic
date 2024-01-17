
module C1BNets

# Neural Network to do Formal Logic
#
# using a model inspired by a single Cortical Column in brain
#
# leverage the scale-free assumption, to preserve fixed (yet really tiny,
# compared to fully-connected) capacity for possible synapses tracked in simu.
#
# mission: to see how close neural-formal-logic can get,
#          toward general / common-sense reasoning
#

export C1BNet, createC1BNet


using StructArrays
using ..Neuronets


function allocAxonSlot(
  cgi::Int, uid::Int, minCap::Int;
  cellsAxonn::Array{Int},
  axons::StructVector{Axon},
  axonEnds::Vector{Int}, # axon groups (by cap according to free-scale assumption)
)
  @assert cgi >= 1
  @assert minCap >= 1
  if cgi > length(axonEnds)
    throw(ArgumentError("requested capacity too large: $minCap vs max preserved $(length(axons.synapses[end]))"))
  end
  axonn = ub = axonEnds[cgi]
  if length(axons.synapses[ub]) < minCap
    # this cap group doesn't have sufficient capacity, resort to the next larger one
    return allocAxonSlot(cgi + 1, uid, minCap;
      cellsAxonn, axons, axonEnds)
  end

  function settleAxon()
    axons.pren[axonn] = uid
    cellsAxonn[uid] = axonn
    return axonn
  end

  # attempt easy path first, try find an unoccupied axon slot and settle with it
  lb = cgi > 1 ? axonEnds[cgi-1] + 1 : 1
  for axonn in lb:ub
    if axons.pren[axonn] == 0
      @assert axons.nsynapse[axonn] == 0
      return settleAxon()
    end
  end

  # all axon slots in this cap group occupied, have to go the hard path
  # migrate the biggest axon within current cap group into next larger group,
  # then settle into its slot

  # locate largest axonn to migrate
  axonn, migLen = lb, axons.nsynapse[lb]
  for n in lb+1:ub
    if axons.nsynapse[n] > migLen
      migLen = axons.nsynapse[n]
      axonn = n
    end
  end

  # allocate the target axon slot in the cap group that next larger
  mig2axonn = allocAxonSlot(cgi + 1, axons.pren[axonn], migLen;
    cellsAxonn, axons, axonEnds)
  # do the migration
  axons.synapses[mig2axonn][1:migLen] = axons.synapses[axonn][1:migLen]
  axons.nsynapse[mig2axonn] = migLen

  # settle into this slot
  axons.nsynapse[axonn] = 0 # empty synapses, as migrated out now
  return settleAxon()
end


struct C1BNet
  cells::StructArray{Neuron}
  integratAxons::StructVector{Axon}
  modulateAxons::StructVector{Axon}
  axonEnds::Vector{Int} # axon groups (by cap according to free-scale assumption)

  potentialConst::Ref{Float64}

  clear::Function # clear its mind 

end


function createC1BNet(
  ncols::Int=1_0000, # total number of Cortical Columns
  nmccs::Int=100, # number of cells per Mini Column
  gamma::Float64=2.2, # scale-free factor
  # expected range of number-of-links per neuron/axon
  nlinks1_min::Int=8, nlinks1_max::Int=10000,
  # unit-capacity of the first/smallest cap group of axon slots
  axoncap1::Int=32,
)
  @assert 1000 <= ncols <= 5_0000
  @assert 50 <= nmccs <= 5000
  @assert 2.0 <= gamma <= 3.0
  @assert 1 <= nlinks1_min <= axoncap1 < nlinks1_max

  cells = StructArray{Neuron}((
    LinearIndices((nmccs, ncols)), # uid
    zeros(Int, (nmccs, ncols)), # integratAxonn
    zeros(Int, (nmccs, ncols)), # modulateAxonn
    zeros(Int8, (nmccs, ncols)), # potential
  ))
  ncells = length(cells)

  potentialConst = Ref{Float64}(0)

  axon_capcnts = let capcnts = Tuple{Int,Int}[],
    cap = axoncap1, k = nlinks1_min, p = 0.0,
    C = ncells / sum(k^(-gamma) for k in nlinks1_min:nlinks1_max)

    while k <= nlinks1_max
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
  axonEnds = cumsum([cnt for (cap, cnt) in axon_capcnts])

  # number of tracked axons
  naxons = axonEnds[end]
  @assert naxons >= ncells
  # total number of tracked synapses
  nlinks = sum(cap * cnt for (cap, cnt) in axon_capcnts)

  # storage for tracked synapses
  integrat_synapses = StructVector{Dendrite}((
    zeros(Int, nlinks), # postn
    zeros(Float32, nlinks), # effi
  ))
  modulate_synapses = StructVector{Dendrite}((
    zeros(Int, nlinks), # postn
    zeros(Float32, nlinks), # effi
  ))

  integratAxons, modulateAxons = let dendrite_begin = 1
    integratAxon_synapses = StructVector{Dendrite}[]
    modulateAxon_synapses = StructVector{Dendrite}[]
    sizehint!(integratAxon_synapses, naxons)
    sizehint!(modulateAxon_synapses, naxons)
    for (cap, cnt) in axon_capcnts
      for _ in 1:cnt
        dendrite_end = dendrite_begin + cap - 1
        push!(integratAxon_synapses, StructVector{Dendrite}((
          integrat_synapses.postn[dendrite_begin:dendrite_end],
          integrat_synapses.effi[dendrite_begin:dendrite_end],
        )))
        push!(modulateAxon_synapses, StructVector{Dendrite}((
          modulate_synapses.postn[dendrite_begin:dendrite_end],
          modulate_synapses.effi[dendrite_begin:dendrite_end],
        )))
        dendrite_begin = dendrite_end + 1
      end
    end
    @assert dendrite_begin == nlinks + 1
    StructVector{Axon}((
      zeros(Int, naxons), # pren
      zeros(Int, naxons), # nsynapse
      integratAxon_synapses, # synapses
    )), StructVector{Axon}((
      zeros(Int, naxons), # pren
      zeros(Int, naxons), # nsynapse
      modulateAxon_synapses, # synapses
    ))
  end

  allocIntegratAxonSlot(uid::Int, minCap::Int=1) = allocAxonSlot(
    1, uid, minCap;
    cellsAxonn=cells.integratAxonn::Array{Int},
    axons=integratAxons,
    axonEnds=axonEnds,
  )
  allocModulateAxonSlot(uid::Int, minCap::Int=1) = allocAxonSlot(
    1, uid, minCap;
    cellsAxonn=cells.modulateAxonn,
    axons=modulateAxons,
    axonEnds=axonEnds,
  )

  function clear()
    cells.potential[:] = [
      r <= 1 ? 1.0 : r^(-gamma) for r in cells.axonn
    ]
    potentialConst[] = 1.0 / sum(cells.potential)
  end

  return C1BNet(
    cells, integratAxons, modulateAxons, axonEnds,
    potentialConst,
    clear,
  )
end


end # module C1BNets 
