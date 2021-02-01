import numpy as np

class MonteCarloLocalization(object):
    """ Sequential Monte Carlo Localization.
    
    Attributes:
    -----------

    x_range : float
        extent in x coordinate
    y_range : float
        extent in y coordinate
    n_particles : int
        number of particles (N)
    """

    def __init__(self, prior_fn, n_particles=200):
        self.prior_fn = prior_fn,
        self.n_particles = n_particles,
        self.init_filter()

    def init_filter(self):
        pass
        #sample = self.prior_fn(self.n_particles)