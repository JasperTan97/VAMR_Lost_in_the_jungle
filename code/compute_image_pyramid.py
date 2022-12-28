import cv2

def computeImagePyramid(img, num_octaves):
    image_pyramid = []
    image_pyramid.append(img)
    for i in range(num_octaves - 1):
        image_pyramid.append(cv2.resize(image_pyramid[i], (0,0), fx = 0.5, fy = 0.5))

    return image_pyramid
