
import numpy
import math
import torch
import torch.utils.data
import torchvision

# addpath('../')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import ptlk


# def rotation():
#     c1 = sympy.Symbol("c1")
#     s1 = sympy.Symbol("s1")
#     c2 = sympy.Symbol("c2")
#     s2 = sympy.Symbol("s2")
#     c3 = sympy.Symbol("c3")
#     s3 = sympy.Symbol("s3")
#
#     Rx = sympy.Matrix([[1, 0, 0], [0, c1, -s1], [0, s1, c1]])
#     Ry = sympy.Matrix([[c2, 0, s2], [0, 1, 0], [-s2, 0, c2]])
#     Rz = sympy.Matrix([[c3, -s3, 0], [s3, c3, 0], [0, 0, 1]])
#
#     print((Rz * Ry * Rx))
#     # Matrix([[c2*c3, -c1*s3 + c3*s1*s2, c1*c3*s2 + s1*s3],\
#     #         [c2*s3,  c1*c3 + s1*s2*s3, c1*s2*s3 - c3*s1],\
#     #         [  -s2,             c2*s1,            c1*c2]])
#
#     # singular case: c2 == 0, s2 == (-1, +1)
#     # --> Rz == identity.
#     # Matrix([[ c2, s1*s2, c1*s2],
#     #         [  0,    c1,   -s1],
#     #         [-s2,     0,     0]])


def rotm2eul(m):
    """ m (3x3, rotation matrix) --> rotation m = Rz*Ry*Rx
    """
    c = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = c < 1e-6
    if not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], c)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], c)
        z = 0

    return numpy.array([x, y, z])


#EOF