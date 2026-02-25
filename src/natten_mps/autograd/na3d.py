from natten_mps.autograd._factory import (
    create_na_autograd_fn,
    create_na_av_autograd_fn,
    create_na_qk_autograd_fn,
)

NeighborhoodAttention3DFunction = create_na_autograd_fn(3)
NA3DQKFunction = create_na_qk_autograd_fn(3)
NA3DAVFunction = create_na_av_autograd_fn(3)
