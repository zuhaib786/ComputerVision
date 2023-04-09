import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
class CheckerBoardPoints:
    '''
    Reference: CV2 Documentation
    '''
    def __init__(self, CHECKERBOARD_SIZE = (6, 7)):
        self.CHECKERBOARD_SIZE = CHECKERBOARD_SIZE
    
    
    def FindPoints(self, filename, visualize = True):
        CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((self.CHECKERBOARD_SIZE[0] * self.CHECKERBOARD_SIZE[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.CHECKERBOARD_SIZE[0], 0:self.CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)

        objpoints = [] # 3D points in real world space
        imgpoints = [] # 2D points in image plane
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.CHECKERBOARD_SIZE, None)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
            imgpoints.append(corners2)
            if visualize:
                img = cv2.drawChessboardCorners(img, self.CHECKERBOARD_SIZE, corners2, ret)
                plt.imshow(img)
                plt.show()
        objpoints = np.array(objpoints)
        imgpoints = np.array(imgpoints)
        _, m, n= objpoints.shape
        objpoints = objpoints.reshape(( m, n))
        
        _, m, _, n = imgpoints.shape
        imgpoints = imgpoints.reshape((m, n))
        objpoints = objpoints[:, :-1]
        return objpoints, imgpoints

# H = CheckerBoardPoints(CHECKERBOARD_SIZE=(6, 7))
# obj, img  = H.FindPoints('Data/C1.jpeg')
# print(obj, img)