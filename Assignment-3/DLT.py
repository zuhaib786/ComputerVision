
import numpy as np
class DirectLinearTransform:
    '''
    Class for solving the equation ||Ax|| = 0
    subject to constration ||x||= 1


    Method involved: Direct Linear Transform
    Using SVD to find the eigen vector corresponding to lowest eigen value

    We use numpy library to perform SVD
    Normalization is done to get stable values and improve the results of DLT
    We can choose to set normalize to False. Default is True
    Create Matrix function is to passed that is a method for finding A given correspondecnes x and y
    '''
    def __init__(self):
        pass

    def normalize2d(self, points):
        '''
        Reference: https://www.cs.brandeis.edu/~cs155/Lecture_06_6.pdf
        '''


        x, y = points[:, 0], points[:, 1]
        mu_x, mu_y = x.mean(), y.mean()
        var_x, var_y = x.var(), y.var()
        s_x, s_y = np.sqrt(2/var_x), np.sqrt(2/var_y)
        transform =  np.matrix([
            [s_x,   0,   -s_x * mu_x],
            [0,   s_y,   -s_y * mu_y],
            [0,     0,              1]
        ])

        # mean, std = np.mean(points, 0), np.std(points)
        # transform = np.array(
        #     [
        #         [std/np.sqrt(2), 0, mean[0]],
        #         [0, std/np.sqrt(2), mean[1]],
        #         [0, 0, 1]
        #     ]
        # )
        # transform =   np.linalg.inv(transform)
        # apply transform
        m, _ = points.shape
        
        points = transform @ np.concatenate(
            [points.T, np.ones((1, m))]
        )
        points = points[0:2].T
        return points, transform
 
    
    def DLT(self,A):
        '''
        Find the lowest non-zero eigen vector using SVD
        '''
        _,S, V = np.linalg.svd(A)
        idx = np.argmin(S)
        return V[idx]
