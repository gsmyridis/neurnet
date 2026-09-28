from collections.abc import Callable

import mlx.core as mx

type MLXLossFunction = Callable[[mx.array, mx.array], mx.array]
