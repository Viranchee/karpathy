## Makemore

### Bigram Model

Updating a 27x27 tensor with integers took:

Jax-Metal: 3 seconds
Pytorch: 45 seconds
Pytorch MPS backend: ~10 minutes
Jax: Not yet tried
MLX: Not yet tried. I do not know how to do this.

I went ahead with Jax-metal and finished the task.
Have yet to do with MLX

But Pytorch is faster at calculating Negative Log Likelihood
16+ minutes JAX vs 40 seconds PyTorch
But 7.3 seconds with Numpy.Log