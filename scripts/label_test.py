import cv2
import numpy as np

image = cv2.imread('D:/01140.png')
image = image[:,:,0]
image_npy = np.load('D:/01140.npy')
image_npy = image_npy[2,:,:]
print(image[:,:,0])