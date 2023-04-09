from Matching import Matching
from Harris import HarrisCornerDetector
from matplotlib import pyplot as plt
import cv2
import os
import numpy as np
from tqdm import tqdm
class AffineModel:
    def __init__(self):
        self.match = Matching()
        self.detector = HarrisCornerDetector()
    def createMatrix(self, points):
        n, _ = points.shape
        A = np.zeros((2*n, 6))
        A[0:2*n:2, 2] = 1
        A[0:2*n:2, 0:2] = points
        A[1:2*n:2, 3:5] = points
        A[1:2*n:2, 5] = 1
        return A
    def createLoad(self, points):
        n, _ = points.shape
        b = np.zeros((2*n , 1))
        b[:2*n:2,0] = points[:, 0]
        b[1:2*n:2,0] = points[:,1]
        return b
    def estimateParams(self, points1, points2):
        A = self.createMatrix(points1)
        b = self.createLoad(points2)
        res = np.linalg.pinv(A.T@A)@(A.T@b)
        ans = np.zeros((2, 3))
        # [a b c]
        # [d e f]
        ans[0 ,:] = res[:3, 0]
        ans[1,:] = res[3:, 0]
        return ans
    def builtinEstimator(self, points1,points2):

        return cv2.estimateAffine2D(points1, points2, )[0]
    def getMatches(self, img1, img2, points1, points2):
        Di= self.match.findNearestMatches(img1, img2, points1, points2)
        points = list(Di.keys())
        expoints  = [Di[point] for point in points]
        points = np.array(points).astype(np.float32)
        expoints = np.array(expoints).astype(np.float32)
        return points, expoints
    def getAffine(self, img1, img2, points1, points2):
        matches = self.getMatches(img1, img2, points1, points2)
        return self.builtinEstimator(matches[0], matches [1])
    def getAffines(self, imageArray, points):
        affines = []
        for ((img1, img2), (points1, points2)) in zip(zip(imageArray[:-1], imageArray[1:]), zip(points[:-1], points[1:])):
            affines.append(self.getAffine(img2, img1, points2, points1))
        return affines
    
    def generatePanorama(self, filename, dim = (400, 400)):
        files = os.listdir(filename)
        files.sort(reverse  = False)
        # files = files[:2]
        imgArray3D = [cv2.imread(filename + '/' + file) for file in files ]
        imgArray3D = [cv2.resize(img, dim, interpolation = cv2.INTER_AREA) for img in imgArray3D]
        imgArray = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in imgArray3D]
        points = [self.detector.getCorners(imgArray[i], K = 50) for i in tqdm(range(len(imgArray)))]
        affines = self.getAffines(imgArray, points)
        result = imgArray3D[-1]
        H = np.eye(3)
        for idx in  range(len(imgArray3D) - 2, -1 , - 1):
            H = affines[idx]
          
            # H = np.linalg.pinv(H)
            img = imgArray3D[idx]
            # tresult = cv2.warpAffine(img, H, dim)
            tresult = cv2.warpAffine(result, H,
                (result.shape[1] + img.shape[1], result.shape[0]))
            # plt.imshow(tresult)
            # plt.show()
            tresult[0:img.shape[0], 0:img.shape[1]] = img
            result = tresult
        result = self.trim(result)
        plt.imshow(result)
        plt.savefig(filename + '/pan.jpg')
        plt.show()

    def trim(self, frame):
        #crop top
        if np.sum(frame[0]) == 0:
            return self.trim(frame[1:])
        #crop bottom
        elif np.sum(frame[-1]) == 0:
            return self.trim(frame[:-1])
        #crop left
        elif np.sum(frame[:,0]) == 0:
            return self.trim(frame[:,1:]) 
        #crop right
        elif np.sum(frame[:,-1]) == 0:
            return self.trim(frame[:,:-1])    
        return frame
A = AffineModel()
A.generatePanorama('Test_Dataset_Assignment_1/Test Dataset')