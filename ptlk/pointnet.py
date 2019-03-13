""" PointNet
    References.
        .. [1] Charles R. Qi, Hao Su, Kaichun Mo and Leonidas J. Guibas,
        "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation",
        CVPR (2017)
"""

import torch

def flatten(x):
    return x.view(x.size(0), -1)

class Flatten(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return flatten(x)


def mlp_layers(nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
    """ [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    layers = []
    last = nch_input
    for i, outp in enumerate(nch_layers):
        if b_shared:
            weights = torch.nn.Conv1d(last, outp, 1)
        else:
            weights = torch.nn.Linear(last, outp)
        layers.append(weights)
        layers.append(torch.nn.BatchNorm1d(outp, momentum=bn_momentum))
        layers.append(torch.nn.ReLU())
        if b_shared == False and dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        last = outp
    return layers

class MLPNet(torch.nn.Module):
    """ Multi-layer perception.
        [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    def __init__(self, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
        super().__init__()
        list_layers = mlp_layers(nch_input, nch_layers, b_shared, bn_momentum, dropout)
        self.layers = torch.nn.Sequential(*list_layers)

    def forward(self, inp):
        out = self.layers(inp)
        return out


def symfn_max(x):
    # [B, K, N] -> [B, K, 1]
    a = torch.nn.functional.max_pool1d(x, x.size(-1))
    #a, _ = torch.max(x, dim=-1, keepdim=True)
    return a

def symfn_avg(x):
    a = torch.nn.functional.avg_pool1d(x, x.size(-1))
    #a = torch.sum(x, dim=-1, keepdim=True) / x.size(-1)
    return a

class TNet(torch.nn.Module):
    """ [B, K, N] -> [B, K, K]
    """
    def __init__(self, K):
        super().__init__()
        # [B, K, N] -> [B, K*K]
        self.mlp1 = torch.nn.Sequential(*mlp_layers(K, [64, 128, 1024], b_shared=True))
        self.mlp2 = torch.nn.Sequential(*mlp_layers(1024, [512, 256], b_shared=False))
        self.lin = torch.nn.Linear(256, K*K)

        for param in self.mlp1.parameters():
            torch.nn.init.constant_(param, 0.0)
        for param in self.mlp2.parameters():
            torch.nn.init.constant_(param, 0.0)
        for param in self.lin.parameters():
            torch.nn.init.constant_(param, 0.0)

    def forward(self, inp):
        K = inp.size(1)
        N = inp.size(2)
        eye = torch.eye(K).unsqueeze(0).to(inp) # [1, K, K]

        x = self.mlp1(inp)
        x = flatten(torch.nn.functional.max_pool1d(x, N))
        x = self.mlp2(x)
        x = self.lin(x)

        x = x.view(-1, K, K)
        x = x + eye
        return x


class PointNet_features(torch.nn.Module):
    def __init__(self, dim_k=1024, use_tnet=False, sym_fn=symfn_max, scale=1):
        super().__init__()
        mlp_h1 = [int(64/scale), int(64/scale)]
        mlp_h2 = [int(64/scale), int(128/scale), int(dim_k/scale)]

        self.h1 = MLPNet(3, mlp_h1, b_shared=True).layers
        self.h2 = MLPNet(mlp_h1[-1], mlp_h2, b_shared=True).layers
        #self.sy = torch.nn.Sequential(torch.nn.MaxPool1d(num_points), Flatten())
        self.sy = sym_fn

        self.tnet1 = TNet(3) if use_tnet else None
        self.tnet2 = TNet(mlp_h1[-1]) if use_tnet else None

        self.t_out_t2 = None
        self.t_out_h1 = None

    def forward(self, points):
        """ points -> features
            [B, N, 3] -> [B, K]
        """
        x = points.transpose(1, 2) # [B, 3, N]
        if self.tnet1:
            t1 = self.tnet1(x)
            x = t1.bmm(x)

        x = self.h1(x)
        if self.tnet2:
            t2 = self.tnet2(x)
            self.t_out_t2 = t2
            x = t2.bmm(x)
        self.t_out_h1 = x # local features

        x = self.h2(x)
        #x = flatten(torch.nn.functional.max_pool1d(x, x.size(-1)))
        x = flatten(self.sy(x))

        #if self.ret_global:
        #    pass
        #else:
        #    # local + global
        #    l0 = self.t_out_h1 # [B, 64, N]
        #    g0 = x # [B, K]
        #    x = torch.cat((l0, g0.unsqueeze(2).repeat(1, 1, num_points)), dim=1)

        return x

class PointNet_classifier(torch.nn.Module):
    def __init__(self, num_c, ptfeat, dim_k):
        super().__init__()
        self.features = ptfeat
        list_layers = mlp_layers(dim_k, [512, 256], b_shared=False, bn_momentum=0.1, dropout=0.0)
        #list_layers = mlp_layers(dim_k, [512, 256], b_shared=False, bn_momentum=0.1, dropout=0.3)
        list_layers.append(torch.nn.Linear(256, num_c))
        self.classifier = torch.nn.Sequential(*list_layers)

    def forward(self, points):
        feat = self.features(points)
        out = self.classifier(feat)
        return out

    def loss(self, out, target, w=0.001):
        loss_c = torch.nn.functional.nll_loss(
            torch.nn.functional.log_softmax(out, dim=1), target, size_average=False)

        t2 = self.features.t_out_t2
        if (t2 is None) or (w == 0):
            return loss_c

        batch = t2.size(0)
        K = t2.size(1)  # [B, K, K]
        I = torch.eye(K).repeat(batch, 1, 1).to(t2)
        A = t2.bmm(t2.transpose(1, 2))
        loss_m = torch.nn.functional.mse_loss(A, I, size_average=False)
        loss = loss_c + w * loss_m
        return loss


#EOF