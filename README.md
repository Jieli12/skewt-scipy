# Skew student $t$ distribution

`Skewt-Scipy` is a Python package for skew student $t$ distribution.

## Skew student $t$ distribution

We employ the definition of the skew-t distribution from Azzalini (2014, Section 4.3.1). Firstly, we need two density functions: $t$ distribution and skew-normal distribution respectively. The $t$ distribution is defined as:

$$
f_{t}(x|\nu) = \frac{\Gamma(\frac{\nu+1}{2})}{\sqrt{\nu\pi}\Gamma(\frac{\nu}{2})}\left(1+\frac{x^{2}}{\nu}\right)^{-\frac{\nu+1}{2}}, \quad x\in \mathbb{R},\quad \nu>0,
$$

where $\nu$ denotes the degree of freedom. The skew-normal distribution is defined as:

$$
f_{sn}(x|\xi,\omega,\alpha) = \frac{2}{\omega}\phi\left(\frac{x-\xi}{\omega}\right)\Phi\left(\alpha \left( \frac{x-\xi}{\omega} \right)\right), \quad x\in \mathbb{R},\quad \omega>0,\quad \alpha\in \mathbb{R},
$$

where $\xi, \omega$ are the location and scale parameters respectively. $\alpha$ is the skewness parameter. $\phi(\cdot)$ and $\Phi(\cdot)$ are the standard normal density function and cumulative distribution function respectively.

The skew-t variable can be defined as:

$$
Z= \frac{Z_{0}}{\sqrt{V}},
$$

where $Z_{0}\sim f_{sn}(x|0,1,\alpha)$
and $V\sim \chi^{2}_{\nu}/\nu$ are independent. Then, the density function of $Z$ is

$$
f_{st}(\alpha,\nu)= 2f_{t}(x|\nu)F_{t}\left(\alpha x\sqrt{\frac{\nu+1}{\nu+x^{2}}}\Big|\nu+1\right),
$$

where $F_{t}(\cdot|\nu+1)$ represent the $t$ cumulative density function with degree of freedom $\nu+1$. As $\alpha\in \mathbb{R}$ and $\nu>0$, then skew-t distribution can degenerate several special distributions.

## Installation

Install via pip with

```bash
python3 -m pip install skewt_scipy
```

## Usage

As the class `skewt` inherits from the class `rv_continuous` of `Scipy`, many methods are available. The shape parameters `a` and `df` represent $\alpha$ and $\nu$ (the degree of freedom).

| Method                                                  | Description                                                |
| ------------------------------------------------------- | ---------------------------------------------------------- |
| `rvs(a, df, loc=0, scale=1, size=1, random_state=None)` | Random variates.                                           |
| `pdf(x, a, df, loc=0, scale=1)`                         | Probability density function.                              |
| `logpdf(x, a, df, loc=0, scale=1)`                      | Log of the probability density function.                   |
| `cdf(x, a, df, loc=0, scale=1)`                         | Cumulative distribution function.                          |
| `logcdf(x, a, df, loc=0, scale=1)`                      | Log of the cumulative distribution function.               |
| `ppf(q, a, df, loc=0, scale=1)`                         | Percent point function (inverse of `cdf`).                 |
| `stats(a, df, loc=0, scale=1, moments='mvsk')`          | Mean('m'), variance('v'), skew('s'), and/or kurtosis('k'). |
| `fit(data)`                                             | Parameter estimates for generic data.                      |

Note that the parameters $\alpha =\pm\infty$ and $\nu=+\infty$ are valid for the above methods.

## Examples

```python
import numpy as np

from skewt_scipy.skewt import skewt

# random number generator
skewt.rvs(a=10, df=6, loc=3, scale=2, size=10)

# probability distribution
x = (np.linspace(-50, 50, 100),)
skewt.pdf(x=x, a=-10, df=6, loc=3, scale=2)

# log of probability distribution
skewt.logpdf(x=x, a=-10, df=6, loc=3, scale=2)

# cumulative distribution
skewt.cdf(x=x, a=8, df=10, loc=3, scale=2)

# log of cumulative distribution
skewt.logcdf(x=x, a=8, df=10, loc=3, scale=2)

# percent point function
skewt.ppf(np.array([0.5, 0.9, 0.99]), a=3, df=6, loc=3, scale=2)

a = 3
df = 5
loc = 3
scale = 2
data = skewt.rvs(a=a, scale=scale, df=df, loc=loc, size=10000, random_state=123)
skewt.fit(data)
skewt.fit(data, fdf=df)  # fixed df
skewt.fit(data, fa=a, fdf=df)  # fixed a and df
```
