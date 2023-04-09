from CheckerBoardPoints import CheckerBoardPoints
from DLT import DirectLinearTransform
import numpy as np
class CameraHomography:
    def __init__(self):
        self.checkerBoard = CheckerBoardPoints(CHECKERBOARD_SIZE=(6, 7))
        self.DLT = DirectLinearTransform()
    def getHomography(self, filename):
        objpoints, imgpoints = self.checkerBoard.FindPoints(filename, visualize=False)

        A, objtransform, imgtransform = self.createMatrix(objpoints, imgpoints)
        H = self.DLT.DLT(A)
        H = H.reshape((3, 3))
        # H = H.T
        return np.linalg.inv(imgtransform)@H@objtransform
        # return np.dot(
        #     np.dot(
        #     np.linalg.inv(objtransform),
        #     H
        #     ),
        #     imgtransform
        # )
    def createMatrix(self, objpoints, imgpoints):
        objpoints, objtransform = self.DLT.normalize2d(objpoints)
        imgpoints, imgtransform = self.DLT.normalize2d(imgpoints)
        # print(objpoints, np.mean(objpoints, axis = 0), np.std(objpoints, axis = 0))
        # print(imgpoints, np.mean(imgpoints, axis = 0), np.std(imgpoints, axis = 0))


        m, n = objpoints.shape
        A = np.zeros((2*m, 9))
        for i in range(m):
            Xi = objpoints[i,0]
            Yi = objpoints[i, 1]
            xi = imgpoints[i, 0]
            yi = imgpoints[i, 1]
            A[2*i, :] = np.array([-Xi, -Yi, -1, 0,0,0, xi*Xi, xi*Yi, xi])
            A[2*i + 1, :] = np.array([0, 0, 0, -Xi, -Yi, -1, yi*Xi, yi*Yi, yi])
            
        return A, objtransform, imgtransform
# H = CameraHomography()
# hom = H.getHomography('Data/C5.jpeg')
# C = CheckerBoardPoints()
# objpoints, imgpoints = C.FindPoints('Data/C5.jpeg', visualize=True)
# m, _ = objpoints.shape
# objpoints = np.concatenate(
#     [
#         objpoints.T,
#         np.ones((1, m))
#     ]
# )

# imgpoints = np.concatenate(
#     [
#         imgpoints.T,
#         np.ones((1, m))
#     ]
# )
# imgpoints = imgpoints.T
# pred = hom@objpoints
# pred = pred.T
# pred = pred/pred[:, -1].reshape((m, 1))
# print(imgpoints - pred)
# print(imgpoints)