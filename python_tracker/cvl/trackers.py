from pyexpat import features
from statistics import stdev
from unittest.mock import patch
import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal import convolve2d as conv2d
from .image_io import crop_patch
from copy import copy
import cv2
import torch
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline

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
        

class MOSSERGBDFtracker:
    def __init__(self, learning_rate=0.1, lam=0.1, deep_extractor = None):
        self.template = None
        self.last_response = None
        self.region = None
        self.bbox = None
        self.region_shape = None
        self.region_center = None
        self.learning_rate = learning_rate
        self.A = None
        self.B = None
        self.M = None
        self.C = None
        self.hann = None
        self.lam = lam
        self.deep_extractor = deep_extractor

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

    def get_normalized_bbox(self, image):
        patch = crop_patch(image, self.bbox)
        patch = patch / 255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        return patch

    def get_channels(self, image):
        features = []
        imagepatch1 = self.get_normalized_patch(image[:,:,0])
        imagepatch2 = self.get_normalized_patch(image[:,:,1])
        imagepatch3 = self.get_normalized_patch(image[:,:,2])
        features.append(imagepatch1)
        features.append(imagepatch2)
        features.append(imagepatch3)
        return features

    def get_deep_features(self, image):
        
        features = []

        imagepatch1 = self.get_normalized_patch(image[:,:,0])
        imagepatch2 = self.get_normalized_patch(image[:,:,1])
        imagepatch3 = self.get_normalized_patch(image[:,:,2])

        imagepatch = np.zeros((imagepatch1.shape[0], imagepatch1.shape[1], 3))
        imagepatch[:,:,0] = imagepatch1
        imagepatch[:,:,1] = imagepatch2
        imagepatch[:,:,2] = imagepatch3

        #upsampled_imagepatch = cv2.resize(imagepatch, (244,244), interpolation=cv2.INTER_CUBIC)

        deep_features = self.deep_extractor(image)

        for f in deep_features:
            feature_map = f.squeeze(0)
            grayscale = torch.sum(feature_map, 0)
            grayscale = grayscale / feature_map.shape[0]
            grayscale = grayscale.detach().cpu().numpy()
            grayscale = cv2.resize(grayscale, (image.shape[1],image.shape[0]), interpolation=cv2.INTER_CUBIC)
            grayscale = self.get_normalized_patch(grayscale)
            features.append(grayscale)
            # cv2.imshow("img",grayscale.detach().cpu().numpy())
            # cv2.waitKey(0)
            # plt.imshow(upsampled_imagepatch)
            # plt.show()
        # fig = plt.figure()
        # fig.add_subplot(1,4,1)
        # plt.imshow(features[0])
        # fig.add_subplot(1,4,2)
        # plt.imshow(features[1])
        # fig.add_subplot(1,4,3)
        # plt.imshow(features[2])
        # fig.add_subplot(1,4,4)
        # plt.imshow(features[3])
        # plt.show()

        # plt.imshow(features[0])
        # plt.show()
        return features

    def get_all_features(self, image):
        channels = self.get_channels(image)
        deep_features = self.get_deep_features(image)
        features = np.concatenate((channels, deep_features))
        return features

    def get_gaussian(self, feature):
        height, width = feature.shape
        x0, y0 = (height // 2, width // 2)
        x, y = np.meshgrid(range(width), range(height))
        stdev = 2
        y = np.exp(-((x-x0)**2+(y-y0)**2)/(2*stdev**2))
        return fft2(y)

    def start(self, image, region):
        self.bbox = copy(region)
        self.region = region.rescale(2, True)
        self.region_center = [self.region.height // 2, self.region.width // 2]
        self.hann = self.get_hanning_window()
        y0, x0 = self.region_center
        x, y = np.meshgrid(range(self.region.width), range(self.region.height))
        
        stdev = 2
        y = np.exp(-((x-x0)**2+(y-y0)**2)/(2*stdev**2))
        self.Y = fft2(y)
        features = self.get_all_features(image)
        
        self.A = []
        self.B = 0
        
        for i, f in enumerate(features):
            X = fft2(f)
            self.A.append(np.multiply(np.conj(self.Y), X))
            self.B += np.multiply(np.conj(X), X)

        
        self.M = []
        self.A = np.array(self.A)
        self.M = np.divide(self.A, self.lam + self.B)

    def detect(self, image):
        features = self.get_all_features(image)
        sums = 0
        for i, f in enumerate(features):
            patch = f
            
            patchf = fft2(patch) #* self.hann
            #M_pad = np.pad(self.M[i], [(16,), (20,)], mode="constant")
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

    def update(self, image, lr=0.9):

        features = self.get_all_features(image)

        B_prev = self.B
        self.B = 0
        for i, f in enumerate(features):
            patch = f
            X = fft2(patch)
            self.A[i] = lr*np.conj(self.Y) * X + (1-lr)*self.A[i]
            self.B += lr*(np.multiply(np.conj(X), (X)))
        self.B += (1-lr)*B_prev
        self.M = np.divide(self.A, self.lam + self.B)
