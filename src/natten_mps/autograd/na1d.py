from natten_mps.autograd._factory import (
    create_na_autograd_fn,
    create_na_av_autograd_fn,
    create_na_qk_autograd_fn,
)

NeighborhoodAttention1DFunction = create_na_autograd_fn(1)
NA1DQKFunction = create_na_qk_autograd_fn(1)
NA1DAVFunction = create_na_av_autograd_fn(1)
