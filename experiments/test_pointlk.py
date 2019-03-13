"""
    Example for testing PointNet-LK.

    No-noise version.
"""

import argparse
import os
import sys
import logging
import numpy
import torch
import torch.utils.data
import torchvision

# addpath('../')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import ptlk


LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def options(argv=None):
    parser = argparse.ArgumentParser(description='PointNet-LK')

    # required.
    parser.add_argument('-o', '--outfile', required=True, type=str,
                        metavar='FILENAME', help='output filename (.csv)')
    parser.add_argument('-i', '--dataset-path', required=True, type=str,
                        metavar='PATH', help='path to the input dataset') # like '/path/to/ModelNet40'
    parser.add_argument('-c', '--categoryfile', required=True, type=str,
                        metavar='PATH', help='path to the categories to be tested') # eg. './sampledata/modelnet40_half1.txt'
    parser.add_argument('-p', '--perturbations', required=True, type=str,
                        metavar='PATH', help='path to the perturbation file') # see. generate_perturbations.py

    # settings for input data
    parser.add_argument('--dataset-type', default='modelnet', choices=['modelnet'],
                        metavar='DATASET', help='dataset type (default: modelnet)')
    parser.add_argument('--format', default='wv', choices=['wv', 'wt'],
                        help='perturbation format (default: wv (twist)) (wt: rotation and translation)') # the output is always in twist format

    # settings for PointNet-LK
    parser.add_argument('--max-iter', default=20, type=int,
                        metavar='N', help='max-iter on LK. (default: 20)')
    parser.add_argument('--pretrained', default='', type=str,
                        metavar='PATH', help='path to trained model file (default: null (no-use))')
    parser.add_argument('--transfer-from', default='', type=str,
                        metavar='PATH', help='path to classifier feature (default: null (no-use))')
    parser.add_argument('--dim-k', default=1024, type=int,
                        metavar='K', help='dim. of the feature vector (default: 1024)')
    parser.add_argument('--symfn', default='max', choices=['max', 'avg'],
                        help='symmetric function (default: max)')
    parser.add_argument('--delta', default=1.0e-2, type=float,
                        metavar='D', help='step size for approx. Jacobian (default: 1.0e-2)')

    # settings for on testing
    parser.add_argument('-l', '--logfile', default='', type=str,
                        metavar='LOGNAME', help='path to logfile (default: null (no logging))')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--device', default='cpu', type=str,
                        metavar='DEVICE', help='use CUDA if available (default: cpu)')

    args = parser.parse_args(argv)
    return args

def main(args):
    # dataset
    testset = get_datasets(args)

    # testing
    act = Action(args)
    run(args, testset, act)


def run(args, testset, action):
    if not torch.cuda.is_available():
        args.device = 'cpu'
    args.device = torch.device(args.device)

    LOGGER.debug('Testing (PID=%d), %s', os.getpid(), args)

    model = action.create_model()
    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
    model.to(args.device)

    # dataloader
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=1, shuffle=False, num_workers=args.workers)

    # testing
    LOGGER.debug('tests, begin')
    action.eval_1(model, testloader, args.device)
    LOGGER.debug('tests, end')


class Action:
    def __init__(self, args):
        self.filename = args.outfile
        # PointNet
        self.transfer_from = args.transfer_from
        self.dim_k = args.dim_k
        self.sym_fn = None
        if args.symfn == 'max':
            self.sym_fn = ptlk.pointnet.symfn_max
        elif args.symfn == 'avg':
            self.sym_fn = ptlk.pointnet.symfn_avg
        # LK
        self.delta = args.delta
        self.max_iter = args.max_iter
        self.xtol = 1.0e-7
        self.p0_zero_mean = True
        self.p1_zero_mean = True

    def create_model(self):
        ptnet = self.create_pointnet_features()
        return self.create_from_pointnet_features(ptnet)

    def create_pointnet_features(self):
        ptnet = ptlk.pointnet.PointNet_features(self.dim_k, use_tnet=False, sym_fn=self.sym_fn)
        if self.transfer_from and os.path.isfile(self.transfer_from):
            ptnet.load_state_dict(torch.load(self.transfer_from, map_location='cpu'))
        return ptnet

    def create_from_pointnet_features(self, ptnet):
        return ptlk.pointlk.PointLK(ptnet, self.delta)

    def eval_1__header(self, fout):
        cols = ['h_w1', 'h_w2', 'h_w3', 'h_v1', 'h_v2', 'h_v3', \
                'g_w1', 'g_w2', 'g_w3', 'g_v1', 'g_v2', 'g_v3'] # h: estimated, g: ground-truth twist vectors
        print(','.join(map(str, cols)), file=fout)
        fout.flush()

    def eval_1__write(self, fout, ig_gt, g_hat):
        x_hat = ptlk.se3.log(g_hat) # --> [-1, 6]
        mx_gt = ptlk.se3.log(ig_gt) # --> [-1, 6]
        for i in range(x_hat.size(0)):
            x_hat1 = x_hat[i] # [6]
            mx_gt1 = mx_gt[i] # [6]
            vals = torch.cat((x_hat1, -mx_gt1)) # [12]
            valn = vals.cpu().numpy().tolist()
            print(','.join(map(str, valn)), file=fout)
        fout.flush()

    def eval_1(self, model, testloader, device):
        model.eval()
        with open(self.filename, 'w') as fout:
            self.eval_1__header(fout)
            with torch.no_grad():
                for i, data in enumerate(testloader):
                    p0, p1, igt = data
                    res = self.do_estimate(p0, p1, model, device) # --> [1, 4, 4]
                    ig_gt = igt.cpu().contiguous().view(-1, 4, 4) # --> [1, 4, 4]
                    g_hat = res.cpu().contiguous().view(-1, 4, 4) # --> [1, 4, 4]

                    dg = g_hat.bmm(ig_gt) # if correct, dg == identity matrix.
                    dx = ptlk.se3.log(dg) # --> [1, 6] (if corerct, dx == zero vector)
                    dn = dx.norm(p=2, dim=1) # --> [1]
                    dm = dn.mean()

                    self.eval_1__write(fout, ig_gt, g_hat)
                    LOGGER.info('test, %d/%d, %f', i, len(testloader), dm)


    def do_estimate(self, p0, p1, model, device):
        p0 = p0.to(device) # template (1, N, 3)
        p1 = p1.to(device) # source (1, M, 3)
        r = ptlk.pointlk.PointLK.do_forward(model, p0, p1, self.max_iter, self.xtol,\
                                            self.p0_zero_mean, self.p1_zero_mean)
        #r = model(p0, p1, self.max_iter)
        est_g = model.g # (1, 4, 4)

        return est_g


def get_datasets(args):

    cinfo = None
    if args.categoryfile:
        #categories = numpy.loadtxt(args.categoryfile, dtype=str, delimiter="\n").tolist()
        categories = [line.rstrip('\n') for line in open(args.categoryfile)]
        categories.sort()
        c_to_idx = {categories[i]: i for i in range(len(categories))}
        cinfo = (categories, c_to_idx)

    perturbations = None
    fmt_trans = False
    if args.perturbations:
        perturbations = numpy.loadtxt(args.perturbations, delimiter=',')
    if args.format == 'wt':
        fmt_trans = True

    if args.dataset_type == 'modelnet':
        transform = torchvision.transforms.Compose([\
                ptlk.data.transforms.Mesh2Points(),\
                ptlk.data.transforms.OnUnitCube(),\
            ])

        testdata = ptlk.data.datasets.ModelNet(args.dataset_path, train=0, transform=transform, classinfo=cinfo)

        testset = ptlk.data.datasets.CADset4tracking_fixed_perturbation(testdata,\
                        perturbations, fmt_trans=fmt_trans)

    return testset


if __name__ == '__main__':
    ARGS = options()

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s:%(name)s, %(asctime)s, %(message)s',
        filename=ARGS.logfile)
    LOGGER.debug('Testing (PID=%d), %s', os.getpid(), ARGS)

    main(ARGS)
    LOGGER.debug('done (PID=%d)', os.getpid())

#EOF