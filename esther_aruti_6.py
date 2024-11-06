'''
Esther Aruti
BME I5000: Medical Imaging and Signal Processing
October 29, 2024
Assignment #6
'''

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.signal

# load in img and crop it
image = Image.open('cell.tif')
img = np.array(image)
g = img[:100,:100]

# generate a gaussian filter in the space domain
kernel_size = 100
sigma = 3

x = np.linspace(- (kernel_size // 2), kernel_size // 2, kernel_size)
y = np.linspace(- (kernel_size // 2), kernel_size // 2, kernel_size)
x, y = np.meshgrid(x, y)
kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
h = kernel / np.sum(kernel)

# apply ft to kernel to turn into freq domain
H = np.fft.fft2(h)
H_s = np.fft.fftshift(h) 

# apply ft to image to bring to freq domain
G = scipy.fft.fft2(g) 

# get b blurred image (convolve with g and h) 
b = scipy.signal.convolve2d(g,h, mode='same')

# get the B (convolved img in freq domain)
B = H * G

# get G (image in frequency domain)
G_unblurred = B / (H)

# get g (the original image)
g_inv = np.fft.ifft2(G_unblurred).astype(float)

# plot original image, kernel, convolved image, and unblurred image
fig, ax = plt.subplots(1, 5, figsize=(21, 6))
ax[0].imshow(g)
ax[0].set_title("Original Image")

ax[1].imshow(h)
ax[1].set_title("Gaussian Kernel in Spatial Domain")

ax[2].imshow(np.abs(np.log(H_s)))
ax[2].set_title("Gaussian Kernel in Frequency Domain")

ax[3].imshow(b)
ax[3].set_title("Blurred Image After Convolving")

ax[4].imshow(g_inv)
ax[4].set_title("Unblurred Image")
plt.show()
