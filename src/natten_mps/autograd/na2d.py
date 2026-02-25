from natten_mps.autograd._factory import (
    create_na_autograd_fn,
    create_na_av_autograd_fn,
    create_na_qk_autograd_fn,
)

NeighborhoodAttention2DFunction = create_na_autograd_fn(2)
NA2DQKFunction = create_na_qk_autograd_fn(2)
NA2DAVFunction = create_na_av_autograd_fn(2)
