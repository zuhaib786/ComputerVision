from Harris import HarrisCornerDetector
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm
sys.setrecursionlimit(10000)
class Matching:
    def __init__(self):
        self.detector = HarrisCornerDetector()
    def findNearestMatches(self, img1, img2, points1, points2):
        diff = [2, 1, 0, -1, -2]
        m, n = img1.shape
        distances = []
        for (x, y) in points1:
            for (i, j) in points2:
                sse = 0
                for dx in diff:
                    for dy in diff:
                        val1, val2 = 0,0 
                        if x + dx >= 0 and x + dx < m and y + dy >=0 and y + dy < n:
                            val1 += img1[x + dx, y + dy]
                        if i + dx >= 0 and i + dx < m and j + dy >= 0 and j + dy < n:
                            val2 += img2[i +dx, j + dy]
                        sse += (val1 - val2)**2
                distances.append([ 0.1 * np.sqrt(sse) + np.abs(x - i) + np.abs(y - j),sse, (y, x), (j, i)])
        distances.sort()
        Di = {}
        used = set()
        for i in distances:
            if (i[2] not in Di.keys()) and (i[3] not in used):
                Di[i[2]] = i[3]
                used.add(i[3])
        return Di           
    def generatePanorama(self, filename, dim = (400, 400), vis =False):
        files = os.listdir(filename)
        files.sort(reverse = False)
        # files = files[1:3]
        imgArray3D = [cv2.imread(filename + '/' + file) for file in files ]
        imgArray3D = [cv2.resize(img, dim, interpolation = cv2.INTER_AREA) for img in imgArray3D]
        imgArray = [cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in imgArray3D]
        print('Finding keypoints')
        points = [self.detector.getCorners(imgArray[i], K = 50) for i in tqdm(range(len(imgArray)))]
        print('Completed')
        homgraphies = self.findHomographies(imgArray, points, vis)
        result = imgArray3D[-1]
        H = np.eye(3)
        for idx in  range(len(imgArray3D) - 2, -1 , - 1):
            H = homgraphies[idx]
            img = imgArray3D[idx]
            tresult = cv2.warpPerspective(result, H,
                (result.shape[1] + img.shape[1], result.shape[0]))
            tresult[0:img.shape[0], 0:img.shape[1]] = img
            result = tresult
        result = self.trim(result)
        plt.imshow(result)
        plt.savefig(filename + '/pan.jpg')
        plt.show()
    def match(self, img1, img2, points1, points2, vis = False):
        
        Di = self.findNearestMatches(img1, img2, points1, points2)
        points = list(Di.keys())
        expoints  = [Di[point] for point in points]

        points = np.array(points)
        expoints = np.array(expoints)
        points = points.astype(np.float32)
        expoints = expoints.astype(np.float32)
        return points, expoints
    def findAffine(self, img1, img2, points1, points2, vis = False):
        matches = self.match(img1, img2, points1, points2, vis = vis)
        return cv2.findHomography(matches[0], matches [1], cv2.RANSAC, 5.0)
    def findHomographies(self, imgArray, points, vis ):
        homographies = []
        for ((img1, img2), (points1, points2)) in zip(zip(imgArray[:-1], imgArray[1:]), zip(points[:-1], points[1:])):
            homographies.append(self.findAffine(img2, img1, points2, points1, vis)[0])        
            return homographies
    def trim(self, img):
        if np.sum(img[0]) == 0:
            return self.trim(img[1:])
        elif np.sum(img[-1]) == 0:
            return self.trim(img[:-1])
        elif np.sum(img[:,0]) == 0:
            return self.trim(img[:,1:]) 
        elif np.sum(img[:,-1]) == 0:
            return self.trim(img[:,:-1])    
        return img
