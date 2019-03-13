""" datasets """

import numpy
import torch
import torch.utils.data

from . import globset
from . import mesh
from .. import so3
from .. import se3


class ModelNet(globset.Globset):
    """ [Princeton ModelNet](http://modelnet.cs.princeton.edu/) """
    def __init__(self, dataset_path, train=1, transform=None, classinfo=None):
        loader = mesh.offread
        if train > 0:
            pattern = 'train/*.off'
        elif train == 0:
            pattern = 'test/*.off'
        else:
            pattern = ['train/*.off', 'test/*.off']
        super().__init__(dataset_path, pattern, loader, transform, classinfo)

class ShapeNet2(globset.Globset):
    """ [ShapeNet](https://www.shapenet.org/) v2 """
    def __init__(self, dataset_path, transform=None, classinfo=None):
        loader = mesh.objread
        pattern = '*/models/model_normalized.obj'
        super().__init__(dataset_path, pattern, loader, transform, classinfo)


class CADset4tracking(torch.utils.data.Dataset):
    def __init__(self, dataset, rigid_transform, source_modifier=None, template_modifier=None):
        self.dataset = dataset
        self.rigid_transform = rigid_transform
        self.source_modifier = source_modifier
        self.template_modifier = template_modifier

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pm, _ = self.dataset[index]
        if self.source_modifier is not None:
            p_ = self.source_modifier(pm)
            p1 = self.rigid_transform(p_)
        else:
            p1 = self.rigid_transform(pm)
        igt = self.rigid_transform.igt

        if self.template_modifier is not None:
            p0 = self.template_modifier(pm)
        else:
            p0 = pm

        # p0: template, p1: source, igt: transform matrix from p0 to p1
        return p0, p1, igt


class CADset4tracking_fixed_perturbation(torch.utils.data.Dataset):
    @staticmethod
    def generate_perturbations(batch_size, mag, randomly=False):
        if randomly:
            amp = torch.rand(batch_size, 1) * mag
        else:
            amp = mag
        x = torch.randn(batch_size, 6)
        x = x / x.norm(p=2, dim=1, keepdim=True) * amp
        return x.numpy()

    @staticmethod
    def generate_rotations(batch_size, mag, randomly=False):
        if randomly:
            amp = torch.rand(batch_size, 1) * mag
        else:
            amp = mag
        w = torch.randn(batch_size, 3)
        w = w / w.norm(p=2, dim=1, keepdim=True) * amp
        v = torch.zeros(batch_size, 3)
        x = torch.cat((w, v), dim=1)
        return x.numpy()

    def __init__(self, dataset, perturbation, source_modifier=None, template_modifier=None,
                 fmt_trans=False):
        self.dataset = dataset
        self.perturbation = numpy.array(perturbation) # twist (len(dataset), 6)
        self.source_modifier = source_modifier
        self.template_modifier = template_modifier
        self.fmt_trans = fmt_trans # twist or (rotation and translation)

    def do_transform(self, p0, x):
        # p0: [N, 3]
        # x: [1, 6]
        if not self.fmt_trans:
            # x: twist-vector
            g = se3.exp(x).to(p0) # [1, 4, 4]
            p1 = se3.transform(g, p0)
            igt = g.squeeze(0) # igt: p0 -> p1
        else:
            # x: rotation and translation
            w = x[:, 0:3]
            q = x[:, 3:6]
            R = so3.exp(w).to(p0) # [1, 3, 3]
            g = torch.zeros(1, 4, 4)
            g[:, 3, 3] = 1
            g[:, 0:3, 0:3] = R # rotation
            g[:, 0:3, 3] = q   # translation
            p1 = se3.transform(g, p0)
            igt = g.squeeze(0) # igt: p0 -> p1
        return p1, igt

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        twist = torch.from_numpy(numpy.array(self.perturbation[index])).contiguous().view(1, 6)
        pm, _ = self.dataset[index]
        x = twist.to(pm)
        if self.source_modifier is not None:
            p_ = self.source_modifier(pm)
            p1, igt = self.do_transform(p_, x)
        else:
            p1, igt = self.do_transform(pm, x)

        if self.template_modifier is not None:
            p0 = self.template_modifier(pm)
        else:
            p0 = pm

        # p0: template, p1: source, igt: transform matrix from p0 to p1
        return p0, p1, igt



#EOF
