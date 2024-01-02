
module BInt8Type

export BInt8

struct BInt8
  value::UInt8

  BInt8(x::UInt8) = new(x)
  BInt8(x::Integer) = new(UInt8(x))
end

Base.zero(::Type{BInt8}) = BInt8(zero(UInt8))

# Conversion to BInt8
Base.convert(::Type{BInt8}, x::Integer) = BInt8(x)
Base.convert(::Type{BInt8}, x::AbstractFloat) = BInt8(round(x))

# Conversion from BInt8
Base.convert(::Type{T}, x::BInt8) where {T<:Integer} = T(x.value)
Base.convert(::Type{T}, x::BInt8) where {T<:AbstractFloat} = T(x.value)

# Promote BInt8 to wider integer types
Base.promote_rule(::Type{BInt8}, ::Type{T}) where {T<:Integer} = T

# Promote BInt8 to floating point types
Base.promote_rule(::Type{BInt8}, ::Type{T}) where {T<:AbstractFloat} = T

# Equality and comparison operations
for op in (:(==), :(<), :(<=), :(>), :(>=))
  @eval Base.$op(b::BInt8, x::Number) = Base.$op(b.value, x)
  @eval Base.$op(x::Number, b::BInt8) = Base.$op(x, b.value)
end

# Addition
function Base.:(+)(a::BInt8, b::BInt8)
  r = UInt16(a.value) + UInt16(b.value)
  return BInt8(UInt8(r > 255 ? 255 : r))
end
function Base.:(+)(a::BInt8, b::T) where {T<:Integer}
  r = T(a.value) + b
  return BInt8(UInt8(clamp(r, 0, 255)))
end
function Base.:(+)(a::T, b::BInt8) where {T<:Integer}
  r = a + T(b.value)
  return BInt8(UInt8(clamp(r, 0, 255)))
end

# Subtraction
function Base.:(-)(a::BInt8, b::BInt8)
  r = Int16(a.value) - Int16(b.value)
  return BInt8(UInt8(r <= 0 ? 0 : r))
end
function Base.:(-)(a::BInt8, b::T) where {T<:Integer}
  r = T(a.value) - b
  return BInt8(UInt8(clamp(r, 0, 255)))
end
function Base.:(-)(a::T, b::BInt8) where {T<:Integer}
  r = a - T(b.value)
  return BInt8(UInt8(clamp(r, 0, 255)))
end

# Multiplication
function Base.:(*)(a::BInt8, b::BInt8)
  r = UInt16(a.value) * UInt16(b.value)
  return BInt8(UInt8(r > 255 ? 255 : r))
end
function Base.:(*)(a::BInt8, b::T) where {T<:Integer}
  r = T(a.value) * b
  return BInt8(UInt8(clamp(r, 0, 255)))
end
function Base.:(*)(a::T, b::BInt8) where {T<:Integer}
  r = a * T(b.value)
  return BInt8(UInt8(clamp(r, 0, 255)))
end


end # module BInt8Type
