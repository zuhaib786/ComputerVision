import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter
import cv2
import matplotlib.pyplot as plt
import os
class HarrisCornerDetector:
    '''
    Harris corener detector
    Derivatives are calculated using sobel filter
    Instead of summing over a window, we use weighted average over a window using gaussian filter
    Corner value is calculated using det(H) - K* (tr[H]**2)
    Value of K is hyperparameter and is taken to be 0.05
    '''
    def __init__(self):
        self.kernels = {}
        # Sobel filters(Averaging along with finding derivatives)
        self.kernels['x'] = np.array(
            [
                [-1,0,1],
                [-2,0,2],
                [-1,0,1]
            ]
        )
        self.kernels['y'] = np.array(
            [
                [1,2,1],
                [0,0,0],
                [-1,-2,-1]

            ]
        )
    def convolve2d(self,img, kernel = 'x'):
        '''
        2d conv operation
        '''
        if kernel not in self.kernels.keys():
            raise ValueError('Kernel not found')
        return convolve2d(img, self.kernels[kernel], mode = 'same')
    def Smoothen(self, img):
        img = gaussian_filter(img, ksize = (3, 3), sigmaX= 1, sigmaY= 1)
        return img
    def HarrisMatrix(self, img, k = 0.05):
        '''
        Calculates Harris corner values
        '''
        I_x = self.convolve2d(img, 'x')
        I_y = self.convolve2d(img, 'y')
        I_xx = gaussian_filter(I_x *I_x, sigma = 1) # Gaussian Weighted average instead of direct avergae
        I_xy = gaussian_filter(I_x*I_y, sigma = 1) # Gaussian Weighted average instead of direct avergae
        I_yy = gaussian_filter(I_y * I_y, sigma = 1) # Gaussian Weighted average instead of direct avergae
        detH = I_xx * I_yy - (I_xy)*(I_xy) #Each pixel value of det(H) - k * [trace(H)]**2
        traceHSq = (I_xx + I_yy)**2
        lambda2 = detH - k*traceHSq
        return lambda2
    def nonMaxSupression(self, points, harrisValues, windowsize = 1, K_max = 50):
        '''
        Perform non-max supression
        8- direction Local optima in a window of size (2*windowsize + 1)
        '''
        ans = []
        m,n = harrisValues.shape
        points.sort(key =lambda x: harrisValues[x[0], x[1]] , reverse = True)
        while len(points) > 0 and len(ans) < K_max:
            ans.append(points[0])
            x,y = points[0]
            tempPts = []
            for pt in points:
                xx,yy = pt
                if np.abs(x - xx)>windowsize or np.abs(y - yy)>windowsize:
                    tempPts.append(pt)
            points = tempPts
        return ans
    def selectTopK(self, points, harrisValues, K = 20):
        '''
        Select the top K points sorted on the bassis of harris corner values
        '''
        points.sort(key =lambda x: harrisValues[x[0], x[1]] , reverse = True)
        return points[:K]
    def selectThreshold(self,corners, thresh = 0.1):
        '''
        Perform thresholding on harris values
        Reject points with harris value less than threhsold
        '''
        return (np.abs(corners) > thresh).astype(np.uint8)
    def getCorners(self, img, K = 50):
        '''
        Final K corners based on which, we will do image stitching
        '''
        img = img.astype(np.float32)
        harrisValues = self.HarrisMatrix(img)
        corners = self.selectThreshold(harrisValues)
        points =[]
        m, n = corners.shape
        for i in range(3, m -3):
            for j in range(3, n - 3):
                if corners[i, j] == 1:
                    points.append([i,j])
        points = self.nonMaxSupression(points, harrisValues, windowsize=6, K_max = K)# 8 way Local Optima
        points = self.selectTopK(points, harrisValues, K )
        points = np.array(points)
        return points
