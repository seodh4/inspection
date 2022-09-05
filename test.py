from ctypes import resize
import sys
import os
import cv2
import numpy as np





# read image

imglist = os.listdir('./data/')

for idx, imgfile in enumerate(imglist):

       img=cv2.imread('./data/'+imgfile)
       img = cv2.resize(img, dsize=(1280, 720), interpolation=cv2.INTER_AREA)
       old_image_height, old_image_width, channels = img.shape

       # create new image of desired size and color (blue) for padding
       new_image_width = 1280
       new_image_height = 768
       color = (128,128,128)
       result = np.full((new_image_height,new_image_width, channels), color, dtype=np.float32)

       # compute center offset
       x_center = (new_image_width - old_image_width) // 2
       y_center = (new_image_height - old_image_height) // 2

       # copy img image into center of result image
       result[y_center:y_center+old_image_height, 
              x_center:x_center+old_image_width] = img

       result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB) 

       # print(result)

       result /= 255

       # print(result)
       np.save('./out/input_' + str(idx) + '.npy', result)
# # view result
# cv2.imshow("result", result)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# save result
# cv2.imwrite("lena_centered.jpg", result)



# # img=cv2.imread('./calibration_set_small/input_1.npy')
# data2 = np.load('./data/1.npy')
# # data2 = data2.transpose((0,1,2))[::-1]
# # print(data2)

# # data2 *= 255
# print(data2)
# cv2.imshow('ffsdf',data2)

# cv2.waitKey(0)
# cv2.destroyAllWindows()
