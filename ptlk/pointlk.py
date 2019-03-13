""" PointLK ver. 2018.07.06.
    using approximated Jacobian by backward-difference.
"""

import numpy
import torch

from . import pointnet
from . import se3, so3, invmat


class PointLK(torch.nn.Module):
    def __init__(self, ptnet, delta=1.0e-2, learn_delta=False):
        super().__init__()
        self.ptnet = ptnet
        self.inverse = invmat.InvMatrix.apply
        self.exp = se3.Exp # [B, 6] -> [B, 4, 4]
        self.transform = se3.transform # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]

        w1 = delta
        w2 = delta
        w3 = delta
        v1 = delta
        v2 = delta
        v3 = delta
        twist = torch.Tensor([w1, w2, w3, v1, v2, v3])
        self.dt = torch.nn.Parameter(twist.view(1, 6), requires_grad=learn_delta)

        # results
        self.last_err = None
        self.g_series = None # for debug purpose
        self.prev_r = None
        self.g = None # estimation result
        self.itr = 0

    @staticmethod
    def rsq(r):
        # |r| should be 0
        z = torch.zeros_like(r)
        return torch.nn.functional.mse_loss(r, z, size_average=False)

    @staticmethod
    def comp(g, igt):
        """ |g*igt - I| (should be 0) """
        assert g.size(0) == igt.size(0)
        assert g.size(1) == igt.size(1) and g.size(1) == 4
        assert g.size(2) == igt.size(2) and g.size(2) == 4
        A = g.matmul(igt)
        I = torch.eye(4).to(A).view(1, 4, 4).expand(A.size(0), 4, 4)
        return torch.nn.functional.mse_loss(A, I, size_average=True) * 16

    @staticmethod
    def do_forward(net, p0, p1, maxiter=10, xtol=1.0e-7, p0_zero_mean=True, p1_zero_mean=True):
        a0 = torch.eye(4).view(1, 4, 4).expand(p0.size(0), 4, 4).to(p0) # [B, 4, 4]
        a1 = torch.eye(4).view(1, 4, 4).expand(p1.size(0), 4, 4).to(p1) # [B, 4, 4]
        if p0_zero_mean:
            p0_m = p0.mean(dim=1) # [B, N, 3] -> [B, 3]
            a0[:, 0:3, 3] = p0_m
            q0 = p0 - p0_m.unsqueeze(1)
        else:
            q0 = p0

        if p1_zero_mean:
            #print(numpy.any(numpy.isnan(p1.numpy())))
            p1_m = p1.mean(dim=1) # [B, N, 3] -> [B, 3]
            a1[:, 0:3, 3] = -p1_m
            q1 = p1 - p1_m.unsqueeze(1)
        else:
            q1 = p1

        r = net(q0, q1, maxiter=maxiter, xtol=xtol)

        if p0_zero_mean or p1_zero_mean:
            #output' = trans(p0_m) * output * trans(-p1_m)
            #        = [I, p0_m;] * [R, t;] * [I, -p1_m;]
            #          [0, 1    ]   [0, 1 ]   [0,  1    ]
            est_g = net.g
            if p0_zero_mean:
                est_g = a0.to(est_g).bmm(est_g)
            if p1_zero_mean:
                est_g = est_g.bmm(a1.to(est_g))
            net.g = est_g

            est_gs = net.g_series # [M, B, 4, 4]
            if p0_zero_mean:
                est_gs = a0.unsqueeze(0).contiguous().to(est_gs).matmul(est_gs)
            if p1_zero_mean:
                est_gs = est_gs.matmul(a1.unsqueeze(0).contiguous().to(est_gs))
            net.g_series = est_gs

        return r

    def forward(self, p0, p1, maxiter=10, xtol=1.0e-7):
        g0 = torch.eye(4).to(p0).view(1, 4, 4).expand(p0.size(0), 4, 4).contiguous()
        r, g, itr = self.iclk(g0, p0, p1, maxiter, xtol)

        self.g = g
        self.itr = itr
        return r

    def update(self, g, dx):
        # [B, 4, 4] x [B, 6] -> [B, 4, 4]
        dg = self.exp(dx)
        return dg.matmul(g)

    def approx_Jic(self, p0, f0, dt):
        # p0: [B, N, 3], Variable
        # f0: [B, K], corresponding feature vector
        # dt: [B, 6], Variable
        # Jk = (ptnet(p(-delta[k], p0)) - f0) / delta[k]

        batch_size = p0.size(0)
        num_points = p0.size(1)

        # compute transforms
        transf = torch.zeros(batch_size, 6, 4, 4).to(p0)
        for b in range(p0.size(0)):
            d = torch.diag(dt[b, :]) # [6, 6]
            D = self.exp(-d) # [6, 4, 4]
            transf[b, :, :, :] = D[:, :, :]
        transf = transf.unsqueeze(2).contiguous()  #   [B, 6, 1, 4, 4]
        p = self.transform(transf, p0.unsqueeze(1)) # x [B, 1, N, 3] -> [B, 6, N, 3]

        #f0 = self.ptnet(p0).unsqueeze(-1) # [B, K, 1]
        f0 = f0.unsqueeze(-1) # [B, K, 1]
        f = self.ptnet(p.view(-1, num_points, 3)).view(batch_size, 6, -1).transpose(1, 2) # [B, K, 6]

        df = f0 - f # [B, K, 6]
        J = df / dt.unsqueeze(1)

        return J

    def iclk(self, g0, p0, p1, maxiter, xtol):
        training = self.ptnet.training
        batch_size = p0.size(0)

        g = g0
        self.g_series = torch.zeros(maxiter+1, *g0.size(), dtype=g0.dtype)
        self.g_series[0] = g0.clone()

        if training:
            # first, update BatchNorm modules
            f0 = self.ptnet(p0)
            f1 = self.ptnet(p1)
        self.ptnet.eval() # and fix them.

        # re-calc. with current modules
        f0 = self.ptnet(p0) # [B, N, 3] -> [B, K]

        # approx. J by finite difference
        dt = self.dt.to(p0).expand(batch_size, 6)
        J = self.approx_Jic(p0, f0, dt)

        self.last_err = None
        itr = -1
        # compute pinv(J) to solve J*x = -r
        try:
            Jt = J.transpose(1, 2) # [B, 6, K]
            H = Jt.bmm(J) # [B, 6, 6]
            B = self.inverse(H)
            pinv = B.bmm(Jt) # [B, 6, K]
        except RuntimeError as err:
            # singular...?
            self.last_err = err
            #print(err)
            # Perhaps we can use MP-inverse, but,...
            # probably, self.dt is way too small...
            f1 = self.ptnet(p1) # [B, N, 3] -> [B, K]
            r = f1 - f0
            self.ptnet.train(training)
            return r, g, itr

        itr = 0
        r = None
        for itr in range(maxiter):
            self.prev_r = r
            p = self.transform(g.unsqueeze(1), p1) # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]
            f = self.ptnet(p) # [B, N, 3] -> [B, K]
            r = f - f0

            dx = -pinv.bmm(r.unsqueeze(-1)).view(batch_size, 6)

            # DEBUG.
            #norm_r = r.norm(p=2, dim=1)
            #print('itr,{},|r|,{}'.format(itr+1, ','.join(map(str, norm_r.data.tolist()))))

            check = dx.norm(p=2, dim=1, keepdim=True).max()
            if float(check) < xtol:
                if itr == 0:
                    self.last_err = 0 # no update.
                break

            g = self.update(g, dx)
            self.g_series[itr+1] = g.clone()

        rep = len(range(itr, maxiter))
        self.g_series[(itr+1):] = g.clone().unsqueeze(0).repeat(rep, 1, 1, 1)

        self.ptnet.train(training)
        return r, g, (itr+1)



#EOF