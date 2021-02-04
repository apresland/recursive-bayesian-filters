import numpy as np
import cv2
from numpy.random import uniform
from scipy.stats import norm, gamma, uniform
from localization import MonteCarloLocalization
from visual import Window

process_noise=(10,0.1) # position, yaw
sensor_noise=10

img_size =600
window = Window(img_size)

tl=[int(img_size/10),int(img_size/10)]
tr=[int(9*img_size/10),int(img_size/10)]
bl=[int(img_size/10),int(9*img_size/10)]
br=[int(9*img_size/10),int(9*img_size/10)]

landmarks = np.array([tl, tr, bl, br])


def motion_fn(x, x_prev):
    ''' Returns velocity and yawrate of the previous increment'''
    velocity = np.linalg.norm(np.array([[x_prev[0], x_prev[1]]])- x ,axis=1)    
    yawrate = np.arctan2(np.array([x[1]-x_prev[1]]), np.array([x_prev[0]-x[0] ]))
    yawrate = -(yawrate-np.pi) if yawrate > 0 else -(np.pi+yawrate)            
    return yawrate, velocity


def measure_fn(x):
    '''  Returns distances to known landmarks'''
    ranges = (np.linalg.norm(landmarks - np.array([[x[0],x[1]]]), axis=1) 
        + (np.random.randn(len(landmarks)) * sensor_noise))
    return ranges


def uniform_prior_fn(N):
    particles = np.empty((N, 3))
    particles[:, 0] = np.random.uniform(0, img_size, size=N)
    particles[:, 1] = np.random.uniform(0, img_size, size=N)
    particles[:, 2] = np.random.uniform(0, 2 * np.pi, size=N)
    particles[:, 2] %= 2 * np.pi
    return particles


filter = MonteCarloLocalization(
    n_particles=200,
    prior_fn=uniform_prior_fn,
    landmarks=landmarks,
    process_noise= process_noise,
    sensor_noise=sensor_noise
)

if __name__ == "__main__":

    x_prev = x_prev =np.array([-1,-1])

    while (1):

        x = window.get_coords()

        if filter.initialized:

            ## predict next state from control
            control = motion_fn(x, x_prev)
            filter.predict(control)

            ## sense position data
            observations = measure_fn(x)
            filter.update(observations)

            ## resample weights 
            filter.resample()

            # visualize state estimate
            window.show(filter)
        else:
            filter.init()

        x_prev = x

        # escape
        result = cv2.waitKey(20)
        if result == 27:
            break