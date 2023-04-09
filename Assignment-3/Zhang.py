import os
from CameraHomography import *
from CheckerBoardPoints import *
from DLT import *
class CallibrateCamera:
    def __init__(self, folder,CHECKERBOARD_SIZE = (6, 7)):
        files = os.listdir(folder)
        self.DLT = DirectLinearTransform()
        self.homographies = []
        self.cam = CameraHomography()
        for file in files:
            # print(folder + '/' + file)
            self.homographies.append(self.cam.getHomography(folder + '/' + file))
        # print(self.homographies)
        
    def createVector(self, i, j, H):
        '''
        Reference: https://www.ipb.uni-bonn.de/html/teaching/photo12-2021/2021-pho1-22-Zhang-calibration.pptx.pdf
        Calculation of vij
        '''
        # print("========================")
        # print(H)
        
        return np.array([
        H[0, i] * H[0, j],
        H[0, i] * H[1, j] + H[1, i] * H[0, j],
        H[2, i] * H[0, j] + H[0, i] * H[2, j],
        H[1, i] * H[1, j],
        H[2, i] * H[1, j] + H[1, i] * H[2, j],
        H[2, i] * H[2, j]
    ])
    def createMatrix(self):
        A = []
        for hom in self.homographies:
            hom = hom.reshape((3, 3))
            A.append(self.createVector(0, 1, hom))
            A.append(self.createVector(0, 0, hom) - self.createVector(1, 1, hom))
        return np.array(A)
    
    def getBMatrix(self):
        A = self.createMatrix()
        m, _ = A.shape
        U, S, V = np.linalg.svd(A)
        idx = np.argmin(S)
        return  V[idx]
    
    def getKMatrix(self):
        B0, B1, B2, B3, B4, B5 = self.getBMatrix()
        B = np.array(
            [
                [B0, B1, B2],
                [B1, B3, B4],
                [B2, B4, B5]
            ]
        )

        #  Cholesky Decomposition
        w = B0 * B2 * B5 - B1**2 * B5 - B0 * B4**2 + 2. * B1 * B3 * B4 - B2 * B3**2
        d = B0 * B2 - B1**2

        # Use Zhang's closed form solution for intrinsic parameters (Zhang, Appendix B, pg. 18)
        v0 = (B[0,1] * B[0,2] - B[0,0] * B[1,2]) / (B[0,0] * B[1,1] - B[0,1] * B[0,1])
        lambda_ = B[2,2] - (B[0,2] * B[0,2] + v0 * (B[0,1] * B[0,2] - B[0,0] * B[1,2])) / B[0,0]
        alpha = np.sqrt(lambda_ / B[0,0])
        beta = np.sqrt(lambda_ * B[0,0] / (B[0,0] * B[1,1] - B[0,1] * B[0,1]))
        gamma = -B[0,1] * alpha * alpha * beta / lambda_
        u0 = gamma * v0 / beta - B[0,2] * alpha * alpha / lambda_

        # Reconstitute intrinsic matrix
        K = np.array([[alpha, gamma, u0],
                    [   0.,  beta, v0],
                    [   0.,    0., 1.]])
        return K
    def reorthogonalize(self, R):
        U, S, V = np.linalg.svd(R)
        return U@V
    def getExtrinsics(self, H, K):
        h0, h1, h2 = H[:, 0], H[:, 1], H[:, 2]
        
        inv_K = np.linalg.inv(K)

        lambda_ = 1. / np.linalg.norm(  inv_K @ h0)
        
        r0 = lambda_ * (inv_K @ h0)
        r1 = lambda_ * (inv_K @h1)
        r2 = np.cross(r0, r1, axis = 0) # cross product as r2 is  _|_ to r1 and r0
        t = lambda_ *(inv_K@h2)

        R = np.vstack((r0.T, r1.T, r2.T)).T
        # print(R.shape, t.shape)

        R = self.reorthogonalize(R)

        E = np.hstack((R, t))
        return E
    def FindAllExtrinsics(self):
        self.extrinsics = []
        K = self.getKMatrix()
        self.K = K
        for hom in self.homographies:
            self.extrinsics.append(self.getExtrinsics(hom, K))
        return

callibrate = CallibrateCamera(folder = 'Data')
# print(callibrate.getKMatrix())
for idx in range(5):
    callibrate.FindAllExtrinsics()
    files = os.listdir('Data')
    img = cv2.imread('Data/' + files[idx])
    pt1 = np.array([[0, 0, 0, 1]])
    pt2 = np.array([[0, 1, 0, 1]])
    pt1 = pt1.T
    pt2 = pt2.T
    K = callibrate.K
    print(K)
    E = callibrate.extrinsics[idx]
    P = K@E
    def calculatePoint(pt):
        pt = P@pt
        pt = pt[:2]/pt[-1]
        pt = pt.astype(np.uint32)
        return (int(pt[0, 0]), int(pt[1, 0]))
    a = 2
    def drawCube():
        pt1 = np.array([[0, 0, 0, 1]])
        pt2 = np.array([[0, a, 0, 1 ]])
        pt3 = np.array([[a, 0, 0, 1]])
        pt4 = np.array([[a, a, 0, 1]])
        pt5 = np.array([[0, 0, -a, 1]])
        pt6 = np.array([[0, a, -a, 1]])
        pt7 = np.array([[a, 0, -a, 1]])
        pt8 = np.array([[a, a, -a, 1]])
        pts = [pt1.T, pt2.T, pt3.T , pt4.T, pt5.T, pt6.T, pt7.T, pt8.T]
        
        pts = [calculatePoint(pt) for pt in pts]
        recs = [
            [0, 1, 3, 2],
            [0, 1, 5, 4],
            [1, 3, 7, 5],
            [0, 2, 6, 4],
            [4, 5, 7, 6],
            [2, 3, 7, 6]
        ]
        
        cv2.fillPoly(img, np.array([[pts[i] for i in recs[0]]]), (0, 0, 255))
        for i in range(1, len(recs)):
            cv2.polylines(img, np.array([[pts[j] for j in recs[i]]]), True, (255, 0, 0), 5)
        # cv2.rectangle(img, pts, )
    drawCube()
    plt.imshow(img)
    plt.savefig('image' + str(idx) + '.jpg')
    plt.show()
    
# pt1 = K@E@pt1
# pt2 = K@E@pt2

# print(callibrate.homographies[0]@pp1)
# pt1 = pt1[:2]/pt1[-1]
# pt1 = pt1.astype(np.uint32)
# pt2 = pt2[:2]/pt2[-1]
# pt2 = pt2.astype(np.uint32)
# pt1 = (pt1[0, 0], pt1[1, 0])
# pt2 = (pt2[0,0], pt2[1,0])
# print(pt1, pt2)
# print(pt2)
# image = cv2.line(img, pt1, pt2, (0, 255, 0), 10)
# plt.imshow(img)
# plt.show()
# H = callibrate.homographies[0]
# print(H/np.linalg.norm(np.linalg.inv(K) @H[:, 0]))
# print(K@E)
