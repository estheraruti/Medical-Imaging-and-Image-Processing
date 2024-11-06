'''
Esther Aruti
BME I5000: Medical Imaging and Signal Processing
November 4, 2024
Assignment #7 : Backprojection
'''

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import skimage as ski
import scipy


img = Image.open('CT_image_today.jpg')

# Dr.Parra lecture code

# attenuation distribution
mu = np.array(img).astype('float64')
mu = np.sum(mu,axis=2)


# radon transform
phi = np.linspace(0,180) # angles
g = ski.transform.radon(mu,theta = phi, circle=False) # radon transform


s_range = np.sqrt(mu.shape[0]**2+mu.shape[1]**2)/2
s = np.linspace(-s_range,s_range,g.shape[0])


# image to freq domain
G = np.fft.fft(g, axis=0)


# number of freq bins
k = G.shape[0]


# make the filter
H = np.abs(np.fft.fftfreq(k)).reshape(-1,1) * np.ones((1, g.shape[1]))
# print(H)


# inverse FT
g_filt = np.real(np.fft.ifft(G*H, axis = 0))


# initialize backproj img
i, j = mu.shape
b = np.zeros((i,j))


# create the meshgrid
x,y = np.meshgrid(np.arange(1, j+1) - j/2, np.arange(1, i+1) - i/2)

# do the filtered backprojection (matlab code from lecture slides)
for i in range(len(phi)):
    # calc s for interpolation
    s_val = x *np.cos(np.radians(phi[i])) - y * np.sin(np.radians(phi[i]))

    # do the interpolation
    interpolate = scipy.interpolate.interp1d(s, g_filt[:,i], bounds_error=False, fill_value=0)
    b += interpolate(s_val)



# plot
fig, ax = plt.subplots(2,2, figsize = (10,10))

ax[0,0].imshow(mu)
ax[0, 0].set_title("Original Image")

ax[0, 1].imshow(g, extent=(phi.min(), phi.max(), s.min(), s.max()))
ax[0, 1].set_title("Radon Transform")

ax[1, 0].imshow(g_filt, extent=(phi.min(), phi.max(), s.min(), s.max()))
ax[1, 0].set_title("Filtered Projections")

ax[1, 1].imshow(b)
ax[1, 1].set_title("Filtered Back Projection")

plt.show()