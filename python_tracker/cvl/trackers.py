from pyexpat import features
from statistics import stdev
from unittest.mock import patch
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import convolve2d as conv2d
from skimage.feature import hog 
from .image_io import crop_patch, crop_patch3d
from .features import colornames_image
from copy import copy
import matplotlib.pyplot as plt
import cv2

class NCCTracker:

    def __init__(self, learning_rate=0.1):
        self.template = None
        self.last_response = None
        self.region = None
        self.region_shape = None
        self.region_center = None
        self.learning_rate = learning_rate

    def get_region(self):# 
        return copy(self.region)

    def get_normalized_patch(self, image):
        region = self.region
        patch = crop_patch(image, region)
        patch = patch / 255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        return patch

    def start(self, image, region):
        assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        self.region = copy(region)
        self.region_shape = (region.height, region.width)
        self.region_center = (region.height // 2, region.width // 2)
        patch = self.get_normalized_patch(image)
        self.template = fft2(patch)

    def detect(self, image):
        assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        patch = self.get_normalized_patch(image)
        
        patchf = fft2(patch)

        responsef = self.template * np.conj(patchf)
        response = ifft2(responsef).real

        r, c = np.unravel_index(np.argmax(response), response.shape)

        # Keep for visualisation
        self.last_response = response

        r_offset = np.mod(r + self.region_center[0], self.region.height) - self.region_center[0]
        c_offset = np.mod(c + self.region_center[1], self.region.width) - self.region_center[1]

        self.region.xpos += c_offset
        self.region.ypos += r_offset

        return self.get_region()

    def update(self, image, lr=0.1):
        assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        patch = self.get_normalized_patch(image)
        patchf = fft2(patch)
        self.template = self.template * (1 - lr) + patchf * lr

class MOSSEtracker:
    def __init__(self, learning_rate=0.1):
        self.template = None
        self.last_response = None
        self.region = None
        self.region_shape = None
        self.region_center = None
        self.learning_rate = learning_rate
        self.A = None
        self.B = None
        self.M = None
        self.C = None
    
    def get_region(self):
        return copy(self.region)

    def get_normalized_patch(self, image):
        region = self.region
        patch = crop_patch(image, region)
        patch = patch / 255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        return patch

    def start(self, image, region):
        self.region = copy(region)
        self.region_center = (region.height // 2, region.width // 2)
        patchnormalized = self.get_normalized_patch(image)
        self.template = fft2(patchnormalized)

        (y0,x0) = self.region_center
        (x,y) = np.meshgrid(range(region.width), range(region.height))
        patch = self.get_normalized_patch(image)
        F = fft2(patch)
        stdev = 5
        c = np.exp(-((x-x0)**2+(y-y0)**2)/(2*stdev**2))
        self.C = fft2(c)

        self.A = np.multiply(np.conj(self.C), F)
        self.B = np.multiply(np.conj(F), F) 

        self.M = np.divide(self.A, self.B)

    def detect(self, image):
        patch = self.get_normalized_patch(image)
        
        patchf = fft2(patch)

        responsef = np.conj(self.M) * patchf
        response = ifft2(responsef).real

        r, c = np.unravel_index(np.argmax(response), response.shape)

        # Keep for visualisation
        self.last_response = response

        #r_offset = np.mod(r + self.region_center[0], self.region.height) - self.region_center[0]
        #c_offset = np.mod(c + self.region_center[1], self.region.width) - self.region_center[1]

        r_offset = r - self.region_center[0]
        c_offset = c - self.region_center[1]

        self.region.xpos += c_offset
        self.region.ypos += r_offset

        return self.get_region()

    def update(self, image, lr=0.8):
        patch = self.get_normalized_patch(image)
        patchf = fft2(patch)
        self.A = lr*np.conj(self.C) * patchf+self.A*(1-lr)
        self.B = lr*np.conj(patchf)*patchf + (1-lr)*self.B
        self.M = np.divide(self.A, self.B)
        

class MOSSERGBtracker:
    def __init__(self, lr=0.8, lam=0.1):
        self.template = None
        self.last_response = None
        self.region = None
        self.bbox = None
        self.region_shape = None
        self.region_center = None
        self.lr = lr
        self.A = None
        self.B = None
        self.M = None
        self.C = None
        self.hann = None
        self.lam = lam
        self.fd = None

    def get_hanning_window(self):
        hy = np.hanning(self.region.height)
        hx = np.hanning(self.region.width)
        return hy.reshape(hy.shape[0], 1) * hx
    
    def get_region(self):
        return copy(self.region)

    def get_bbox(self):# 
        return copy(self.bbox)

    def get_normalized_patch(self, image):
        patch = crop_patch(image, self.region)
        patch = patch / 255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        return patch

    def get_normalized_patch3d(self, image):
        patch = crop_patch3d(image, self.region)
        patch = patch / 255
        patch -= np.mean(patch, axis=(0,1), keepdims=True)  
        patch /= np.std(patch, axis=(0,1), keepdims=True)
        return patch

    def get_normalized_bbox(self, image):
        patch = crop_patch(image, self.bbox)
        patch = patch / 255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        return patch

    def hog_features(self, image):
        patch = self.get_normalized_patch3d(image)
        fd, hog_image = hog(patch, pixels_per_cell=(8,8), cells_per_block=(3,3), feature_vector=False, visualize=True, channel_axis=-1)
        return fd.reshape(fd.shape[0], fd.shape[1], -1)

    def get_features(self, image):
        features = []
        features.append(image[:,:,0])
        features.append(image[:,:,1])
        features.append(image[:,:,2])
        return features

    def start(self, image, region):
        self.bbox = copy(region)
        self.region = region.rescale(2.5, True)
        self.region_center = [self.region.height // 2, self.region.width // 2]
        self.hann = self.get_hanning_window()

        y0, x0 = self.region_center
        x, y = np.meshgrid(range(self.region.width), range(self.region.height))
        
        stdev = 2
        y = np.exp(-((x-x0)**2+(y-y0)**2)/(2*stdev**2))
        self.Y = fft2(y)
                
        self.A = []
        self.B = 0
       
        fd = self.hog_features(image)

        for i in range(fd.shape[2]):
            feature = cv2.resize(fd[:,:,i], (self.region.width, self.region.height))
            X = fft2(feature)
            self.A.append(np.multiply(np.conj(self.Y), X))
            self.B += np.multiply(np.conj(X), X)


        self.M = []
        self.A = np.array(self.A)
        self.M = np.divide(self.A, self.lam + self.B)

    def detect(self, image):
        sums = 0
        fd = self.hog_features(image)
        self.fd = fd

        for i in range(fd.shape[2]):
            feature = cv2.resize(fd[:,:,i], (self.region.width, self.region.height))
            patchf = fft2(feature)
            responsef = np.conj(self.M[i]) * patchf # Convolution to match filter with image patch
            response = ifft2(responsef).real
            sums += response

        r, c = np.unravel_index(np.argmax(sums), sums.shape)

        # Keep for visualisation
        self.last_response = sums

        r_offset = r - self.region_center[0]
        c_offset = c - self.region_center[1]

        self.region.xpos += c_offset
        self.region.ypos += r_offset
        self.bbox.xpos += c_offset
        self.bbox.ypos += r_offset

        if (self.bbox.xpos < 0 or self.bbox.xpos + self.bbox.width > image.shape[1] ):
            self.region.xpos -= c_offset
            self.bbox.xpos -= c_offset

        if (self.bbox.ypos < 0 or self.bbox.ypos + self.bbox.height > image.shape[0] ):
            self.region.ypos -= r_offset
            self.bbox.ypos -= r_offset


        return self.get_bbox()

    def update(self):
        fd = self.fd
        B_prev = self.B
        self.B = 0
        for i in range(fd.shape[2]):
            feature = cv2.resize(fd[:,:,i], (self.region.width, self.region.height))
            X = fft2(feature)
            self.A[i] = self.lr*np.conj(self.Y) * X + (1-self.lr)*self.A[i]
            self.B += self.lr*(np.multiply(np.conj(X), (X)))
            i += 1
        self.B += (1-self.lr)*B_prev
        self.M = np.divide(self.A, self.lam + self.B)
