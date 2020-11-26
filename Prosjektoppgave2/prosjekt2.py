import numpy as np
import matplotlib.pyplot as plt

from min_err_rate import min_err_rate

# convert image to array with RGB values
img = plt.imread('Bilde1.png')
img = img[:, :, :3]
img_org = img
print(img.shape)
plt.imshow(img)
plt.show()

sum = np.sum(img, axis=2)
sum[sum == 0] = 1
img[:, :, 0] /= sum
img[:, :, 1] /= sum
img[:, :, 2] /= sum
# img = img[:, :, :2]
plt.imshow(img)
plt.show()

chili = img[200:400, 100:150]
red = img[100:175, 240:350]
green = img[300:420, 230:380]

# plt.imshow(chili)
# plt.show()
# plt.imshow(green)
# plt.show()

print(chili.shape)
print(red.shape)
print(green.shape)

segmentation = min_err_rate(img_org, chili, red, green)
plt.imshow(segmentation)
plt.show()
