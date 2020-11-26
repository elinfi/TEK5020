import cv2
import numpy as np
import matplotlib.pyplot as plt

from min_err_rate import min_err_rate

# convert image to array with RGB values
img = plt.imread('Bilde2.png')
img = img[:, :, :3]

# normalize RGB values
sum = np.sum(img, axis=2)
sum[sum == 0] = 1
img[:, :, 0] /= sum
img[:, :, 1] /= sum
img[:, :, 2] /= sum

img2 = plt.imread('Bilde3.png')
img2 = img2[:, :, :3]

# normalize RGB values
sum2 = np.sum(img2, axis=2)
sum2[sum2 == 0] = 1
img2[:, :, 0] /= sum2
img2[:, :, 1] /= sum2
img2[:, :, 2] /= sum2

plt.imshow(img2)
plt.show()

# cv2.imshow('image', img2)
# cv2.waitKey()
# cv2.destroyAllWindows()

blue = img[400:580, 300:500]
red = img[210:400, 800:1025]
floor = img[650:800, 500:800]

#
segmentation = min_err_rate(img2, blue, red, floor)
plt.imshow(segmentation)
plt.show()
