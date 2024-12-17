
'''
Esther Aruti
BME I5000: Medical Imaging and Signal Processing
December 12, 2024
Assignment #10: Filtering + Edge Detectors
'''


import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import filters
from skimage import feature
from PIL import Image
import scipy
import skimage.morphology



image = Image.open('blood1.tif')
img = np.array(image)


### MAKING MY OWN EDGE DETECTOR

# create a denoising kernel
smooth_kernel = np.array([[1, 2, 1],
                          [2, 12, 2], # changing the center value intensifies the smoothing
                          [1, 2, 1]])
smooth_kernel = smooth_kernel / 18

# Convolve smoothing filter with image to reduce noise
smooth_img = scipy.signal.convolve2d(img, smooth_kernel, mode='same', boundary='wrap')

# make a sharpening kernel
sharp_kernel = np.array([[-1, -1, -1],
                         [-1, 8, -1],
                         [-1, -1, -1]])

# Convolve filter with smoothed image to detect and enhance edges
sharp_img = scipy.signal.convolve2d(smooth_img, sharp_kernel, mode='same', boundary='wrap')

# Do thresholding
thresholded_img = sharp_img > filters.threshold_otsu(sharp_img)


# create kernel fro morphology operation
close_kernel = skimage.morphology.square(2)
# do closing
closed_img = skimage.morphology.erosion(thresholded_img, close_kernel)

my_label_img, my_num_labels = skimage.measure.label(closed_img, return_num=True)
my_sizes = np.bincount(my_label_img.ravel())[1:] 
my_sizes = my_sizes[(my_sizes > 30)]





### COMPARE TO SKIMAGE CANNY EDGE DETECTOR
canny_edges = feature.canny(img)

# do thresholding
canny_thresh_img = canny_edges > skimage.filters.threshold_otsu(canny_edges)

# create kernel for morphology operation
open_kernel = skimage.morphology.square(4)
# do open
canny_open_img = skimage.morphology.dilation(canny_thresh_img, open_kernel)

# create kernel for morphology operation
close_kernel = skimage.morphology.square(6)
# do closing
canny_closed_img = skimage.morphology.erosion(canny_open_img, close_kernel) # morphological operation only does so much to fix the double edges


canny_label_img, canny_num_labels = skimage.measure.label(canny_closed_img, return_num=True)
canny_sizes = np.bincount(canny_label_img.ravel())[1:] 
canny_sizes = canny_sizes[(canny_sizes > 20)]





### COMPARE TO SOBEL EDGE DETECTOR
sobel_edges = filters.sobel(img)

# do thresholding
sobel_thresh_img = sobel_edges > skimage.filters.threshold_otsu(sobel_edges)


# create kernel fro morphology operation
open_kernel = skimage.morphology.square(1)
# do open
sobel_open_img = skimage.morphology.dilation(sobel_thresh_img, open_kernel)

# create kernel fro morphology operation
close_kernel = skimage.morphology.square(2)
# do closing
sobel_closed_img = skimage.morphology.erosion(sobel_open_img, close_kernel)


sobel_label_img, sobel_num_labels = skimage.measure.label(sobel_closed_img, return_num=True)
sobel_sizes = np.bincount(sobel_label_img.ravel())[1:] 
sobel_sizes = sobel_sizes[(sobel_sizes > 20)]





### MEASURE SIZE OF CELLS

my_avg = np.mean(my_sizes)
print('My Mean: ', my_avg)

my_std = np.std(my_sizes)
print('My Standard Deviation: ', my_std)

my_max = np.max(my_sizes)
print("My Max: ", my_max)

my_min = np.min(my_sizes)
print("My Min: ", my_min)

canny_avg = np.mean(canny_sizes)
canny_std = np.std(canny_sizes)
canny_min = np.min(canny_sizes)
canny_max = np.max(canny_sizes)
print("Canny Mean: ", canny_avg)
print("Canny Standard Deviation: ", canny_std)
print("Canny Min: ", canny_min)
print("Canny Max: ", canny_max)


sobel_avg = np.mean(sobel_sizes)
sobel_std = np.std(sobel_sizes)
sobel_min = np.min(sobel_sizes)
sobel_max = np.max(sobel_sizes)
print("Sobel Mean: ", sobel_avg)
print("Sobel Standard Deviation: ", sobel_std)
print("Sobel Min: ", sobel_min)
print("Sobel Max: ", sobel_max)








### PLOT RESULTS

fig, ax = plt.subplots(3, 4, figsize=(20, 15))

# My edge detector
ax[0, 0].imshow(img)
ax[0, 0].set_title('Original Image')
ax[0, 0].axis('off')

ax[0, 1].imshow(sharp_img)
ax[0, 1].set_title('Image After Sharpening and Smoothing')
ax[0, 1].axis('off')

ax[0, 2].imshow(thresholded_img)
ax[0, 2].set_title('Thresholded Image')
ax[0, 2].axis('off')

ax[0, 3].imshow(my_label_img)
ax[0, 3].set_title('Labeled Image')
ax[0, 3].axis('off')

# Canny edge detector
ax[1, 0].imshow(img)
ax[1, 0].set_title('Original Image')
ax[1, 0].axis('off')

ax[1, 1].imshow(canny_edges)
ax[1, 1].set_title('Canny Edges')
ax[1, 1].axis('off')

ax[1, 2].imshow(canny_thresh_img)
ax[1, 2].set_title('Canny Thresholded Image')
ax[1, 2].axis('off')

ax[1, 3].imshow(canny_label_img)
ax[1, 3].set_title('Canny Labeled Image')
ax[1, 3].axis('off')

# Sobel edge detector
ax[2, 0].imshow(img)
ax[2, 0].set_title('Original Image')
ax[2, 0].axis('off')

ax[2, 1].imshow(sobel_edges)
ax[2, 1].set_title('Sobel Edges')
ax[2, 1].axis('off')

ax[2, 2].imshow(sobel_thresh_img)
ax[2, 2].set_title('Sobel Thresholded Image')
ax[2, 2].axis('off')

ax[2, 3].imshow(sobel_label_img)
ax[2, 3].set_title('Sobel Labeled Image')
ax[2, 3].axis('off')

plt.tight_layout()
plt.show()





# plot avg circumference, standard deviation, and mean, max, min of cells for all three methods
fig_stats, ax_stats = plt.subplots(2, 2, figsize=(12, 10))

# Plot average sizes
avg_values = [my_avg, canny_avg, sobel_avg]
ax_stats[0, 0].bar(['My Method', 'Canny', 'Sobel'], avg_values, color='blue')
ax_stats[0, 0].set_title('Average Cell Size')
ax_stats[0, 0].set_ylabel('Average Size (pixels)')

# Plot standard deviations
std_values = [my_std, canny_std, sobel_std]
ax_stats[0, 1].bar(['My Method', 'Canny', 'Sobel'], std_values, color='red')
ax_stats[0, 1].set_title('Standard Deviation of Cell Size')
ax_stats[0, 1].set_ylabel('Standard Deviation (pixels)')

# Plot minimum sizes
min_values = [my_min, canny_min, sobel_min]
ax_stats[1, 0].bar(['My Method', 'Canny', 'Sobel'], min_values, color='green')
ax_stats[1, 0].set_title('Minimum Cell Size')
ax_stats[1, 0].set_ylabel('Size (pixels)')

# Plot maximum sizes
max_values = [my_max, canny_max, sobel_max]
ax_stats[1, 1].bar(['My Method', 'Canny', 'Sobel'], max_values, color='purple')
ax_stats[1, 1].set_title('Maximum Cell Size')
ax_stats[1, 1].set_ylabel('Size (pixels)')

plt.tight_layout()
plt.show()
