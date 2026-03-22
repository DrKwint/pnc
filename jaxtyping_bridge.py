import typing

if typing.TYPE_CHECKING:
    Float = typing.Any
    Int = typing.Any
    Bool = typing.Any
    Array = typing.Any
    f32 = typing.Any
    f64 = typing.Any
else:
    from jaxtyping import Array, Float, Int, Bool, Float32 as f32, Float64 as f64
