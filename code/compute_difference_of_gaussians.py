import cv2
import numpy as np

def computeDifferenceOfGaussians(blurred_images):
    num_octaves = len(blurred_images)
    dogs = []

    for i, img in enumerate(blurred_images):
        dog = np.zeros(img.shape - np.array([0, 0, 1]))
        num_dogs_per_octave = dog.shape[2]
        for dog_idx in range(num_dogs_per_octave):
            dog[:, :, dog_idx] = np.abs( img[:, :, dog_idx+1] - img[:, :, dog_idx] )

        dogs.append(dog)

    return dogs


