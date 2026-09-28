import mlx.core as mx


def normal_like(array: mx.array, mean: float, std: float) -> mx.array:
    """Return a normal initializer sample matching ``array``."""
    return mx.random.normal(
        array.shape,
        dtype=array.dtype,
        loc=mean,
        scale=std,
    )
