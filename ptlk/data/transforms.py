""" gives some transform methods for 3d points """
import math

import numpy as np
import torch
import torch.utils.data

from . import mesh
from .. import so3
from .. import se3


class Mesh2Points:
    def __init__(self):
        pass

    def __call__(self, mesh):
        mesh = mesh.clone()
        v = mesh.vertex_array
        return torch.from_numpy(v).type(dtype=torch.float)

class OnUnitSphere:
    def __init__(self, zero_mean=False):
        self.zero_mean = zero_mean

    def __call__(self, tensor):
        if self.zero_mean:
            m = tensor.mean(dim=0, keepdim=True) # [N, D] -> [1, D]
            v = tensor - m
        else:
            v = tensor
        nn = v.norm(p=2, dim=1) # [N, D] -> [N]
        nmax = torch.max(nn)
        return v / nmax

class OnUnitCube:
    def __init__(self):
        pass

    def method1(self, tensor):
        m = tensor.mean(dim=0, keepdim=True) # [N, D] -> [1, D]
        v = tensor - m
        s = torch.max(v.abs())
        v = v / s * 0.5
        return v

    def method2(self, tensor):
        c = torch.max(tensor, dim=0)[0] - torch.min(tensor, dim=0)[0] # [N, D] -> [D]
        s = torch.max(c) # -> scalar
        v = tensor / s
        return v - v.mean(dim=0, keepdim=True)

    def __call__(self, tensor):
        #return self.method1(tensor)
        return self.method2(tensor)


class Resampler:
    """ [N, D] -> [M, D] """
    def __init__(self, num):
        self.num = num

    def __call__(self, tensor):
        num_points, dim_p = tensor.size()
        out = torch.zeros(self.num, dim_p).to(tensor)

        selected = 0
        while selected < self.num:
            remainder = self.num - selected
            idx = torch.randperm(num_points)
            sel = min(remainder, num_points)
            val = tensor[idx[:sel]]
            out[selected:(selected + sel)] = val
            selected += sel
        return out

class RandomTranslate:
    def __init__(self, mag=None, randomly=True):
        self.mag = 1.0 if mag is None else mag
        self.randomly = randomly
        self.igt = None

    def __call__(self, tensor):
        # tensor: [N, 3]
        amp = torch.rand(1) if self.randomly else 1.0
        t = torch.randn(1, 3).to(tensor)
        t = t / t.norm(p=2, dim=1, keepdim=True) * amp * self.mag

        g = torch.eye(4).to(tensor)
        g[0:3, 3] = t[0, :]
        self.igt = g # [4, 4]

        p1 = tensor + t
        return p1

class RandomRotator:
    def __init__(self, mag=None, randomly=True):
        self.mag = math.pi if mag is None else mag
        self.randomly = randomly
        self.igt = None

    def __call__(self, tensor):
        # tensor: [N, 3]
        amp = torch.rand(1) if self.randomly else 1.0
        w = torch.randn(1, 3)
        w = w / w.norm(p=2, dim=1, keepdim=True) * amp * self.mag

        g = so3.exp(w).to(tensor) # [1, 3, 3]
        self.igt = g.squeeze(0) # [3, 3]

        p1 = so3.transform(g, tensor) # [1, 3, 3] x [N, 3] -> [N, 3]
        return p1

class RandomRotatorZ:
    def __init__(self):
        self.mag = 2 * math.pi

    def __call__(self, tensor):
        # tensor: [N, 3]
        w = torch.Tensor([0, 0, 1]).view(1, 3) * torch.rand(1) * self.mag

        g = so3.exp(w).to(tensor) # [1, 3, 3]

        p1 = so3.transform(g, tensor)
        return p1

class RandomJitter:
    """ generate perturbations """
    def __init__(self, scale=0.01, clip=0.05):
        self.scale = scale
        self.clip = clip
        self.e = None

    def jitter(self, tensor):
        noise = torch.zeros_like(tensor).to(tensor) # [N, 3]
        noise.normal_(0, self.scale)
        noise.clamp_(-self.clip, self.clip)
        self.e = noise
        return tensor.add(noise)

    def __call__(self, tensor):
        return self.jitter(tensor)


class RandomTransformSE3:
    """ rigid motion """
    def __init__(self, mag=1, mag_randomly=False):
        self.mag = mag
        self.randomly = mag_randomly

        self.gt = None
        self.igt = None

    def generate_transform(self):
        # return: a twist-vector
        amp = self.mag
        if self.randomly:
            amp = torch.rand(1, 1) * self.mag
        x = torch.randn(1, 6)
        x = x / x.norm(p=2, dim=1, keepdim=True) * amp

        return x # [1, 6]

    def apply_transform(self, p0, x):
        # p0: [N, 3]
        # x: [1, 6]
        g = se3.exp(x).to(p0)   # [1, 4, 4]
        gt = se3.exp(-x).to(p0) # [1, 4, 4]

        p1 = se3.transform(g, p0)
        self.gt = gt.squeeze(0) #  gt: p1 -> p0
        self.igt = g.squeeze(0) # igt: p0 -> p1
        return p1

    def transform(self, tensor):
        x = self.generate_transform()
        return self.apply_transform(tensor, x)

    def __call__(self, tensor):
        return self.transform(tensor)



#EOF