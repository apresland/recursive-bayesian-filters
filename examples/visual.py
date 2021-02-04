import cv2
import numpy as np

x =np.array([-1,-1])

def mouseCallback(event, px, py, flags,null):
    global x
    x =np.array([px,py])

window_name = "Monte Carlo Localization"
cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouseCallback)

class Window(object):

    def __init__(self, img_size):
        self.img_size=img_size

    def get_coords(self):
        global x
        return x

    def show(self, filter):

        # create base image
        img = np.zeros((self.img_size, self.img_size, 3), np.uint8)

        # add landmarks
        for landmark in filter.landmarks:
            cv2.circle(img, tuple(landmark),10,(255,0,0),-1)
        
        # add particles:
        for particle in filter.particles:
            cv2.circle(img, tuple((int(particle[0]),int(particle[1]))),3,(255,128,255),1)

        # add state estiamte
        mean,variance = filter.estimate()
        cv2.circle(img, tuple((int(mean[0]),int(mean[1]))),3,(0,0,255),-1)

        # show image
        cv2.imshow(window_name, img)