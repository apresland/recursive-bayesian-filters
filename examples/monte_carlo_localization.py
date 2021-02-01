import numpy as np
from numpy.random import uniform
from localization import MonteCarloLocalization

def uniform_prior(N):
    xs = np.array([0,800])
    ys = np.array([0,600])
    particles = np.empty((N, 2))
    particles[:, 0] = uniform(xs[0], xs[1], size=N)
    particles[:, 1] = uniform(ys[0], ys[1], size=N)
    return particles

prior_fn = uniform_prior(200)

def example_mcl():
    # create filter
    filter = MonteCarloLocalization(
        prior_fn=prior_fn,
        n_particles=200)

if __name__ == "__main__":

    print("+++ running localization example +++")
    example_mcl()