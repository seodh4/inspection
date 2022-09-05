from ctypes import resize
import sys
import os
import cv2
import numpy as np


data2 = np.load('./calibration_set_small/1.npy')
# data2 = data2.transpose((0,1,2))[::-1]
# print(data2)

# data2 *= 255
print(data2)
cv2.imshow('ffsdf',data2)

cv2.waitKey(0)
cv2.destroyAllWindows()