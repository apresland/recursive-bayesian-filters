import numpy as np
import scipy as scipy

class MonteCarloLocalization(object):

    def __init__(self, n_particles, prior_fn, landmarks, motion_fn, measure_fn, sensor_noise, process_noise):
        self.n_particles = n_particles
        self.prior_fn = prior_fn
        self.particles =self.prior_fn(self.n_particles)
        self.landmarks =landmarks
        self.weights = np.array([1.0]*n_particles)
        self.motion_fn =motion_fn
        self.measure_fn =measure_fn
        self.sensor_noise =sensor_noise
        self.process_noise =process_noise

    def update(self):
        control = self.motion_fn()
        self.motion_update(u=control)
        observations = self.measure_fn()
        self.sensor_update(z=observations, R=self.sensor_noise)
        self.resample()

    def motion_update(self, u):
        dt =1.0
        dist =(u[1] * dt) + (np.random.randn(self.n_particles) * self.process_noise)
        self.particles[:, 0] += np.cos(u[0]) * dist
        self.particles[:, 1] += np.sin(u[0]) * dist

    def sensor_update(self, z, R):
        self.weights.fill(1.)
        for i, landmark in enumerate(self.landmarks):
            distance=np.power((self.particles[:,0] - landmark[0])**2 +(self.particles[:,1] - landmark[1])**2, 0.5)
            self.weights *= scipy.stats.norm(distance, R).pdf(z[i])
            self.weights += 1.e-300 # avoid round-off to zero
            self.weights /= sum(self.weights)

    def resample(self):
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
        self.particles[:] = self.particles[indexes]
        self.weights[:] = self.weights[indexes]
        self.weights /= np.sum(self.weights)