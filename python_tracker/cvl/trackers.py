from statistics import stdev
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from .image_io import crop_patch
from copy import copy
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
        self.region_shape = (region.height, region.width)
        self.region_center = (region.height // 2, region.width // 2)
        patchnormalized = self.get_normalized_patch(image)
        self.template = fft2(patchnormalized)

        (y0,x0) = self.region_center
        (x,y) = np.meshgrid(range(region.width), range(region.height))
        patch = crop_patch(image, region)
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
        