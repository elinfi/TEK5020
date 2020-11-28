import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as patches

from min_err_rate import min_err_rate

# convert image to array with RGB values
img = plt.imread('../Datafiles/Bilde2.png')
img = img[:, :, :3]
img2 = plt.imread('../Datafiles/Bilde3.png')
img2 = img2[:, :, :3]

# Create figure and axes
fig,ax = plt.subplots(1)

# Display the image
ax.imshow(img)

# Create a Rectangle patch
rect1 = patches.Rectangle((300, 350), 200, 200, linewidth=1, edgecolor='black',
                          facecolor='none')
rect2 = patches.Rectangle((800, 210), 200, 190, linewidth=1, edgecolor='black',
                          facecolor='none')
rect3 = patches.Rectangle((500, 650), 300, 150, linewidth=1, edgecolor='black',
                          facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect1)
ax.add_patch(rect2)
ax.add_patch(rect3)

plt.axis('off')
plt.title('Markerte treningsområder', size=16)
plt.show()

# create train data
blue = img[350:450, 300:500]
red = img[210:400, 800:1000]
floor = img[650:800, 500:800]

# plot segmentation without normalization
segmentation = min_err_rate(img2, blue, red, floor)
plt.imshow(segmentation)
plt.axis('off')
plt.title('Segmentering uten normalisering', size=16)
plt.show()

# normalize RGB values
sum = np.sum(img, axis=2)
sum[sum == 0] = 1
img[:, :, 0] /= sum
img[:, :, 1] /= sum
img[:, :, 2] /= sum

sum2 = np.sum(img2, axis=2)
sum2[sum2 == 0] = 1
img2[:, :, 0] /= sum2
img2[:, :, 1] /= sum2
img2[:, :, 2] /= sum2

img = img[:, :, :2]
img2 = img2[:, :, :2]

# create train data of normalized image
blue = img[350:450, 300:500]
red = img[210:400, 800:1000]
floor = img[650:800, 500:800]

# plot segmentation with normalization
segmentation = min_err_rate(img2, blue, red, floor)
plt.imshow(segmentation)
plt.axis('off')
plt.title('Segmentering med normalisering', size=16)
plt.show()
