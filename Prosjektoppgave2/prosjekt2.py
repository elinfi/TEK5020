import numpy as np
import matplotlib.pyplot as plt

from min_err_rate import min_err_rate

# convert image to array with RGB values
img = plt.imread('Bilde1.png')
img = img[:, :, :3]
plt.imshow(img)
plt.close()

# normalize RGB values
sum = np.sum(img, axis=2)
sum[sum == 0] = 1
img[:, :, 0] /= sum
img[:, :, 1] /= sum
img[:, :, 2] /= sum
# img = img[:, :, :2]
plt.imshow(img)
plt.close()

# create train data
chili = img[200:400, 100:150]
red = img[100:175, 240:350]
green = img[300:420, 230:380]

# plot segmenation
segmentation = min_err_rate(img, chili, red, green)
plt.imshow(segmentation)
plt.show()
