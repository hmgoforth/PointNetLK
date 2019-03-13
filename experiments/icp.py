""" ICP algorithm

    References:
    (ICP)
    [1] Paul J. Besl and Neil D. McKay,
        "A method for registration of 3-D shapes",
        PAMI Vol. 14, Issue 2, pp. 239-256, 1992.
    (SVD)
    [2] K. S. Arun, T. S. Huang and S. D. Blostein,
        "Least-Squares Fitting of Two 3-D Point Sets",
        PAMI Vol. 9, Issue 5, pp.698--700, 1987
"""
import numpy as np
from scipy.spatial import KDTree

def _icp_find_rigid_transform(p_from, p_target):
    A, B = np.copy(p_from), np.copy(p_target)

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    A -= centroid_A
    B -= centroid_B

    H = np.dot(A.T, B)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = np.dot(Vt.T, U.T)

    t = np.dot(-R, centroid_A) + centroid_B

    return R, t

def _icp_Rt_to_matrix(R, t):
    # matrix M = [R, t; 0, 1]
    Rt = np.concatenate((R, np.expand_dims(t.T, axis=-1)), axis=1)
    a = np.concatenate((np.zeros_like(t), np.ones(1)))
    M = np.concatenate((Rt, np.expand_dims(a, axis=0)), axis=0)
    return M

class ICP:
    """ Estimate a rigid-body transform g such that:
        p0 = g.p1
    """
    def __init__(self, p0, p1):
        """ p0.shape == (N, 3)
            p1.shape == (N, 3)
        """
        self.p0 = p0
        self.p1 = p1
        leafsize = 1000
        self.nearest = KDTree(self.p0, leafsize=leafsize)
        self.g_series = None

    def compute(self, max_iter):
        ftol = 1.0e-7
        dim_k = self.p0.shape[1]
        g = np.eye(dim_k + 1, dtype=self.p0.dtype)
        p = np.copy(self.p1)

        self.g_series = np.zeros((max_iter + 1, dim_k + 1, dim_k + 1), dtype=g.dtype)
        self.g_series[0, :, :] = g

        itr = -1
        for itr in range(max_iter):
            neighbor_idx = self.nearest.query(p)[1]
            targets = self.p0[neighbor_idx]
            R, t = _icp_find_rigid_transform(p, targets)

            new_p = np.dot(R, p.T).T + t
            if np.sum(np.abs(p - new_p)) < ftol:
                break

            p = np.copy(new_p)
            dg = _icp_Rt_to_matrix(R, t)
            new_g = np.dot(dg, g)
            g = np.copy(new_g)
            self.g_series[itr + 1, :, :] = g

        self.g_series[(itr+1):, :, :] = g

        return g, p, (itr + 1)



def icp_test():
    from math import sin, cos
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    Y, X = np.mgrid[0:100:5, 0:100:5]
    Z = Y ** 2 + X ** 2
    A = np.vstack([Y.reshape(-1), X.reshape(-1), Z.reshape(-1)]).T

    R = np.array([
        [cos(-0.279), -sin(-0.279), 0],
        [sin(-0.279), cos(-0.279), 0],
        [0, 0, 1]
    ])
    #R = np.eye(3)

    t = np.array([5.0, 20.0, 10.0])
    #t = np.array([0.0, 0.0, 0.0])

    B = np.dot(R, A.T).T + t
    A = A.astype(B.dtype)

    icp = ICP(A, B)
    matrix, points, itr = icp.compute(10)

    print(itr)
    print(icp.g_series)
    print(icp.g_series[itr])
    print(matrix)
    print(R.T)
    print(np.dot(-R.T, t))

    fig = plt.figure()
    #ax = Axes3D(fig)
    ax = fig.add_subplot(111, projection='3d')

    ax.set_label("x - axis")
    ax.set_label("y - axis")
    ax.set_label("z - axis")

    ax.plot(A[:,1], A[:,0], A[:,2], "o", color="#cccccc", ms=4, mew=0.5)
    ax.plot(points[:,1], points[:,0], points[:,2], "o", color="#00cccc", ms=4, mew=0.5)
    ax.plot(B[:,0], B[:,1], B[:,2], "o", color="#ff0000", ms=4, mew=0.5)

    plt.show()

if __name__ == '__main__':
    icp_test()

#EOF