The repository contains serval experiments involving the gumbel-softmax.

gumbel_softmax.py: the implementation of gumbel softmax.
In line 34-48, we generate random variables via gumbel softmax.

```
original prob:
 tensor([[0.3000, 0.6000, 0.1000],
        [0.7000, 0.2000, 0.1000]])
self   :
 tensor([[0.3035, 0.5980, 0.0985],
        [0.6986, 0.2016, 0.0998]])
offical:
 tensor([[0.2991, 0.5987, 0.1023],
        [0.6997, 0.2010, 0.0992]])
```
gumbel_softmax_vae.py: train vae_gumbel_vae

gumbel_softmax_vae_generation.py: An attempt was made to generate images using he trained vae_gumbel_softmax.

# references
1. Jang, E., Gu, S., & Poole, B. (2016, November). Categorical Reparameterization with Gumbel-Softmax. In International Conference on Learning Representations.

2. https://sassafras13.github.io/GumbelSoftmax/

3. https://neptune.ai/blog/gumbel-softmax-loss-function-guide-how-to-implement-it-in-pytorch