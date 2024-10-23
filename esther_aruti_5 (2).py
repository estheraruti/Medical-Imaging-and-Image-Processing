'''
Esther Aruti
BME I5000: Medical Imaging and Signal Processing
October 22, 2024
Assignment #5: Radon Transform
'''

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')  # Backend for dynamic updates
import numpy as np
from PIL import Image
from scipy.ndimage import rotate
from skimage.transform import radon

def radon_transform(image, theta):
    
    m, n = image.shape

    # get the distance to the line
    r = int(np.ceil(np.sqrt(m**2 + n**2)))
    pad_image = np.pad(image, [(r, r), (r, r)], mode='constant') # chatgpt help to prevent cutting off image

    # make empty sinogram
    sinogram = np.zeros((2 * r, len(theta)))

    # for each angle rotate the padded image
    for i, angle in enumerate(theta):
        rotated_image = rotate(pad_image, angle, reshape=False)

        # get the sum of the values of the rotated image
        projection = np.sum(rotated_image, axis=0)
        
        # add to sinogram
        sinogram[:, i] = projection[:sinogram.shape[0]]

    return sinogram

# Load the image in grayscale
original_img = Image.open('CT_image_today.jpg').convert('L')
Img = np.array(original_img)
Img = Img[::5, ::5]  # downsample for speed


# create empty theta
theta = np.linspace(0., 180., max(Img.shape), endpoint=False)
diagonal = int(np.ceil(np.sqrt(Img.shape[0]**2 + Img.shape[1]**2)))

plt.figure()

# plot original image
plt.subplot(1, 3, 1)
plt.imshow(Img, cmap='gray')
plt.title('Original Image')

# do skimage's radon function for comparison
radon_output = radon(np.array(Img), theta=theta)

plt.subplot(1, 3, 3)
plt.imshow(radon_output, aspect='equal', cmap='gray')
plt.title('Radon Output of Full Image')

# make empt cumulative sinogram with zeros
cumulative_sinogram = np.zeros((2 * diagonal, len(theta)))

# Looping over every pixel in the image
sinograms = []  # list to store individual sinograms
for i in range(Img.shape[0]): # NICK'S CODE
    for j in range(Img.shape[1]):
        # Skip black pixels 
        if Img[i, j] == 0:  
            continue 

        # Create an image for the current pixel
        pixel_img = np.zeros_like(Img)
        pixel_img[i, j] = Img[i, j]  # set current pixel to its brightness
        
        # calc radon transform
        sinogram = radon_transform(pixel_img, theta)
        sinograms.append(sinogram)  # store sinogram
        
        # update the cumulative sinogram
        cumulative_sinogram += sinogram

        # display the updated cumulative sinogram
        plt.subplot(1, 3, 2)
        plt.imshow(cumulative_sinogram, aspect='equal', cmap = 'gray')
        plt.title('Cumulative Sinogram')
        
        # Display the image update
        plt.subplot(1, 3, 1)
        plt.imshow(Img, cmap='gray')
        plt.title(f'Processing Pixel [{i}, {j}]')

        # Adding a small delay to allow matplotlib to draw the updated figure
        plt.pause(0.01)  # NICK'S CODE, Adjusted for smoother animation
        plt.draw()



# Final plot
plt.show()




