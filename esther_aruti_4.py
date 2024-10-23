'''
Esther Aruti
BME I5000: Medical Imaging and Signal Processing
October 8, 2024
Assignment #4: Counting Rice Kernels
'''

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import skimage as ski
from scipy.optimize import fmin
import numpy as np
import skimage.filters
import skimage.measure


img = ski.io.imread('rice.png')
print(img.shape)




def myplane(p, dims): # lecture slides
    [X, Y] = np.meshgrid(np.linspace(0, 1, dims[0]), np.linspace(0, 1, dims[1]))
    return p[0] + p[1]*X + p[2]*Y + p[3]*X**2 + p[4]*Y**2 + p[5]*X*Y # changed to parabola


def myerror(p,Z): # lecture Nick's code
    return np.sum(np.sum((Z-myplane(p,np.shape(Z)))**2))


def myplanefit(Z): # lecture slides
    poptimal = fmin(myerror, [np.mean(Z), 0, 0, 0, 0, 0], args=(Z,)) # changed to parabola
    return myplane(poptimal, np.shape(Z))




# detrend the image
detrended_img = img - myplanefit(img)

# thresholding using Otsu
binary_img = detrended_img > skimage.filters.threshold_otsu(detrended_img)

# Label connected components
label_img, num_labels = skimage.measure.label(binary_img, return_num=True)

# create kernel
kernel1 = ski.morphology.square(5)

# erosion
erosion = ski.morphology.erosion(label_img, kernel1)

# create kernel
kernel2 = ski.morphology.square(4)

# dilation
dilation = ski.morphology.dilation(erosion, kernel2)

# Calculate sizes (area) of each component
sizes = np.bincount(dilation.ravel())[1:]  # chatgpt recommended bincount to find area





# Display the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Detrended Image')
plt.imshow(detrended_img, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Binary Image with Labels')
plt.imshow(dilation, cmap='nipy_spectral')
plt.axis('off')

plt.tight_layout()
plt.show()

# remove size 0
sizes = [size for size in sizes if size != 0]

# remove largest 3
sizes = sorted(sizes)[:-3] 

# Print the sizes of the rice kernels
for i, size in enumerate(sizes):
    print(f"Rice Kernel {i + 1}: Size = {size} pixels")





# print # of rice kernels
print("Number of Rice Kernels: ", len(sizes))

# get the average size
print("Average Rice Kernel Size: ", np.mean(sizes))

print("Max Rice Kernel Size: ", np.max(sizes))
print("Min Rice Kernel Size: ", np.min(sizes))

# get the standard deviation
print("Standard Deviation of Rice Kernel Size: ", np.std(sizes))
