"""
Author         : Jie Li, Innovision IP Ltd and School of Mathematics, Statistics and Actuarial Science, University of Kent.
Date           : 2024-01-20 23:30:09
Last Revision  : 2024-01-22 13:42:06
Last Author    : Jie Li
File Path      : /skewt_scipy/test/test.py
Description    :








Copyright (c) 2024, Jie Li, jl725@kent.ac.uk
All Rights Reserved.
"""
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
