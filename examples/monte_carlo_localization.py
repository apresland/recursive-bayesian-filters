import numpy as np
import cv2
from numpy.random import uniform
from localization import MonteCarloLocalization

window_height =800
window_width =800
window_name ="Monte Carlo Localization"

def uniform_prior(N):
    xs = np.array([0,800])
    ys = np.array([0,600])
    particles = np.empty((N, 2))
    particles[:, 0] = uniform(xs[0], xs[1], size=N)
    particles[:, 1] = uniform(ys[0], ys[1], size=N)
    return particles

prior_fn = uniform_prior(200)

def localization():

    # create filter
    filter = MonteCarloLocalization(
        prior_fn=prior_fn,
        n_particles=200)

    # blank image
    img = np.zeros((window_height, window_width,3), np.uint8)

    # create window
    cv2.namedWindow(window_name)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    print("+++ running localization example +++")
    localization()