# Infinity SGD

Python implementation for AAAI-20 submission: Infinity Learning: Learning Markov Chains from Aggregate Steady-State Observations.

# Usage

To use our infinity-SGD function as a common differentiable PyTorch function. Please

```python
import ddmp

# generate a batched transition rate matrices in PyTorch Tensor
rate_matrices = None

# common forwarding
distributions = ddmp.stdy_dist(rate_matrices, c=0.001, dkargs=dict(rvdist=(stats.geom, 0.1)))

# some loss function
l = loss_function(distributions)

# common backwarding (backward function is embedded for ddmp.stdy_dist)
l.backward()
```

- `c`

  The uniformization offset $\epsilon$ in the paper.

- `dkargs`

  Arguments used to define sampling process for Russian Roulette in the paper.

  Commonly, it should have a single key `rvdist`, and a tuple as its value.

  The first element of the tuple defines a random number generator. The following elements are arguments to create the random number generator.

  `scipy.stats.geom` is geometric sampling provided by scipy library, we can create a random process holder by `rng = scipy.stats.geom(0.1)`, then in our implementation, we will call `rng.rvs()` to sample a random variable from defined random distribution.

# Reproduce Experiments

To run all experiments for the paper, you just need to run `sh/share.sh`. It includes all settings.

```bash
for ((i = 1; i <= 8; ++i)); do
	sh/share.sh ml0${i}
done
```

# MKL Round-off

Because of different float number round-off between different machines, you may not recover exactly the same values in our paper, but you should achieve similar result which does not influence the conclusion.

# Future Update

We have a more efficient implementaion (also update linear solver to support CUDA according to recent update of PyTorch 1.2). Since it is related to some ongoing future works, we may publish it later next year.
