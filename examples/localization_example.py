import numpy as np
import cv2
from numpy.random import uniform
from scipy.stats import norm, gamma, uniform
from localization import MonteCarloLocalization

img_size =500
x =np.array([-1,-1])
x_prev =np.array([-1,-1])

sensor_noise=50
process_noise=50

def measure_fn():
    global x
    ranges = (np.linalg.norm(landmarks - np.array([[x[0],x[1]]]), axis=1) 
        + (np.random.randn(len(landmarks)) * sensor_noise))
    return ranges


def motion_fn():
    global x
    global x_prev
    distance = np.linalg.norm(np.array([[x_prev[0], x_prev[1]]])- x ,axis=1)    
    rotation = np.arctan2(np.array([x[1]-x_prev[1]]), np.array([x_prev[0]-x[0] ]))
    rotation = -(rotation-np.pi) if rotation > 0 else -(np.pi+rotation)            
    return rotation, distance


def uniform_prior_fn(N):
    particles = np.empty((N, 2))
    particles[:, 0] = np.random.uniform(0, img_size, size=N)
    particles[:, 1] = np.random.uniform(0, img_size, size=N)
    return particles


def mouseCallback(event, px, py, flags,null):
    global x
    x =np.array([px,py])


window_name ="Monte Carlo Localization"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouseCallback)
landmarks = np.array([[100,100], [400,100], [100,400], [400,400]])

filter = MonteCarloLocalization(
    n_particles=200,
    prior_fn=uniform_prior_fn,
    landmarks=landmarks,
    motion_fn=motion_fn,
    measure_fn=measure_fn,
    sensor_noise= sensor_noise,
    process_noise=process_noise
)

if __name__ == "__main__":

    while (1):

        if x_prev[0] > 0:
            filter.update()
        x_prev = x

        # create base image
        img = np.zeros((img_size, img_size, 3), np.uint8)

        # add landmarks
        for landmark in filter.landmarks:
            cv2.circle(img, tuple(landmark),10,(255,0,0),-1)
        
        # add particles:
        for particle in filter.particles:
            cv2.circle(img,tuple((int(particle[0]),int(particle[1]))),1,(255,255,255),-1)

        # show image
        cv2.imshow(window_name, img)

        # check for escape
        result = cv2.waitKey(20)
        if result == 27:
            break

    cv2.destroyAllWindows()