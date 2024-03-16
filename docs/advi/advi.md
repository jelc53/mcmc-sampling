---
sketchTitle: Variational Inference
sketchAuthor: "Julian Cooper, 30th Oct. 2023"
sketchPublishDate: "2023-09-15"
articleTitle: Automatic Differentiation Variational Inference
articleAuthor: Alp Kucukelbir
articlePublishDate: "2017-01-01"
category: Statistics
bannerImage: /imgs/neuromancer_headset.png
description: "Variational Inference is used to approximate joint posterior distributions. This sketch summarizes the method, its pros and cons, and existing implementations."
---

|     |     |
| --- | --- |  
| **Title** | **Automatic Differentiation Variational Inference** |  
| Author(s) | Alp Kucukelbir, Dustin Tran, Rajesh Ranganath, Andrew Gelman and David Blei |  
| Date published | January, 2017 |  
|     |     |   


### Introduction

Variational Inference, Laplace Approximation and Integrated Nested Laplace Approximation belong to a family of techniques that seek to approximate an intractable posterior distribution with a simpler, tractable distribution. In this sketch we focus on the practical implementation of Variational Inference (VI). 

The first significant publication on VI applied to statistical inference was written in 1999 by Michael I. Jordan, Zoubin Ghahramani, Tommi S. Jaakkola and Lawrence K. Saul. Jordan et al introduced the idea that if one could find a well-chosen family of distributions, then inference could be reframed as an optimization problem where we search for the parameters (i.e. a specific member of this family) that minimize the Kullback-Leibler (KL) divergence between the true posterior and the proposed distribution.

While not the original author, Andrew Gelman made significant contributions in exploring the limitations of VI and enabling its practical usage by combining the technique with Auto-Differentiation software libraries. Similarly, Gelman played a major role in the development of the No U-Turn Sampler (NUTS), including its implementation in the Stan programming language, which popularized and made practical Hamiltonian Monte Carlo (HMC) sampling.
   

### Motivation

Variational Inference is faster than traditional sampling methods ...


Speed!! but concerns about accuracy ... 

How do we know it works ... need to confirm with MCMC test, some tests you czn perform (ref)


### Math intuition

Toy example problem ...

Possible approches to solve for posterior ...

Why ELBO is a natural lower bound ...

Put it together ...


### Stan's ADVI algorithm

First end-to-end algo that automated VI. 
Particularly autograd and proposal dist!

- Pick a proposal distribution

- Rewrite ELBO expectation as an approx. by sampling and taking the average (LLN)

- For each optimization step (theta_l):

  - Sample L times from current proposal distribution (with theta_l)

  - Use samples to compute approx. ELBO (our loss) and store computational graph

  - Estimate gradient of negative ELBO with backprop / autograd

  - Increment gradient-based optimizer (adam) to find updated dist. params (theta_l + 1)

  - Stop when converged to dist. params that minimize negative ELBO


Result is optimal params for proposal distribution that minimizes KL divergence with posterior

Fast alternate version: xx


### Code implementations

Short example ... tensorflow

```python
import numpy as np

def main():
    # print statement to console
    print("Our code looks great!")

if __name__ == "__main__":
    main()
```

Table or just description of the different implementations ...
Stan, PyMC, Pyro, NumPyro, Edward2


### References
- original paper 
- stochastic paper
- recap on variaitonal ifnerence
- how do we know it works

