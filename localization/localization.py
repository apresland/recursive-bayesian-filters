import numpy as np
import scipy as scipy

class MonteCarloLocalization(object):

    def __init__(self, n_particles, prior_fn, landmarks, process_noise, sensor_noise):
        self.initialized =False
        self.n_particles = n_particles
        self.prior_fn = prior_fn
        self.landmarks =landmarks
        self.process_noise =process_noise
        self.sensor_noise =sensor_noise

    # initialize particles with prior distribution
    def init(self):
        self.particles =self.prior_fn(self.n_particles)
        self.weights = np.array([1.0]*self.n_particles)
        self.initialized = True

    # predict next pose from motion control inputs
    def predict(self, u):
        dt =1.0
        dist =(u[1] * dt) + (np.random.randn(self.n_particles) * self.process_noise[0])
        self.particles[:, 0] += np.cos(u[0]) * dist
        self.particles[:, 1] += np.sin(u[0]) * dist
        self.particles[:, 2] += u[0] + (np.random.randn(self.n_particles) * self.process_noise[1])
        self.particles[:, 2] %= 2 * np.pi

    # update particle probability from sensor inputs
    def update(self, z):
        self.weights.fill(1.)
        for i, landmark in enumerate(self.landmarks):
            distance=np.power((self.particles[:,0] - landmark[0])**2 +(self.particles[:,1] - landmark[1])**2, 0.5)
            self.weights *= scipy.stats.norm(distance, self.sensor_noise).pdf(z[i])
            self.weights += 1.e-300 # avoid round-off to zero
            self.weights /= sum(self.weights)

    # estimate state after update 
    def estimate(self):
        positions = self.particles[:, 0:2]
        mean = np.average(positions, weights=self.weights, axis=0)
        variance  = np.average((positions - mean)**2, weights=self.weights, axis=0)
        return mean, variance

    # resampling (systematic resampling) 
    def resample(self):
        indexes = self.systematic_resample()
        self.particles[:] = self.particles[indexes]
        self.weights[:] = self.weights[indexes]
        self.weights /= np.sum(self.weights)

    #
    def systematic_resample(self):
        positions = (np.arange(self.n_particles) + np.random.random()) / self.n_particles
        indexes = np.zeros(self.n_particles, 'i')
        cumulative_sum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < self.n_particles and j<self.n_particles:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes