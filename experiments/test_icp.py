"""
    Example for testing ICP.

    No-noise version.

    This ICP-test is very slow. use faster ones like Matlab or C++...
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

import icp

LOGGER = logging.getLogger(__name__)
LOGGER.addHandler(logging.NullHandler())


def options(argv=None):
    parser = argparse.ArgumentParser(description='ICP')

    # required.
    parser.add_argument('-o', '--outfile', required=True, type=str,
                        metavar='FILENAME', help='output filename (.csv)')
    parser.add_argument('-i', '--dataset-path', required=True, type=str,
                        metavar='PATH', help='path to the input dataset')
    parser.add_argument('-c', '--categoryfile', required=True, type=str,
                        metavar='PATH', help='path to the categories to be tested')
    parser.add_argument('-p', '--perturbations', required=True, type=str,
                        metavar='PATH', help='path to the perturbations')

    # settings for input data
    parser.add_argument('--dataset-type', default='modelnet', choices=['modelnet'],
                        metavar='DATASET', help='dataset type (default: modelnet)')
    parser.add_argument('--format', default='wv', choices=['wv', 'wt'],
                        help='perturbation format (default: wv (twist)) (wt: rotation and translation)') # the output is always in twist format

    # settings for ICP
    parser.add_argument('--max-iter', default=20, type=int,
                        metavar='N', help='max-iter on ICP. (default: 20)')

    # settings for on testing
    parser.add_argument('-l', '--logfile', default='', type=str,
                        metavar='LOGNAME', help='path to logfile (default: null (no logging))')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')

    args = parser.parse_args(argv)
    return args

def main(args):
    # dataset
    testset = get_datasets(args)

    # testing
    act = Action(args)
    run(args, testset, act)


def run(args, testset, action):
    LOGGER.debug('Testing (PID=%d), %s', os.getpid(), args)

    sys.setrecursionlimit(20000)

    # dataloader
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=1, shuffle=False, num_workers=args.workers)

    # testing
    LOGGER.debug('tests, begin')
    action.eval_1(testloader)
    LOGGER.debug('tests, end')


class Action:
    def __init__(self, args):
        self.filename = args.outfile
        # ICP
        self.max_iter = args.max_iter

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

    def eval_1(self, testloader):
        with open(self.filename, 'w') as fout:
            self.eval_1__header(fout)
            with torch.no_grad():
                for i, data in enumerate(testloader):
                    p0, p1, igt = data
                    res = self.do_estimate(p0, p1) # [1, N, 3] x [1, M, 3] --> [1, 4, 4]
                    ig_gt = igt.cpu().contiguous().view(-1, 4, 4) # --> [1, 4, 4]
                    g_hat = res.cpu().contiguous().view(-1, 4, 4) # --> [1, 4, 4]

                    dg = g_hat.bmm(ig_gt) # if correct, dg == identity matrix.
                    dx = ptlk.se3.log(dg) # --> [1, 6] (if corerct, dx == zero vector)
                    dn = dx.norm(p=2, dim=1) # --> [1]
                    dm = dn.mean()

                    self.eval_1__write(fout, ig_gt, g_hat)
                    LOGGER.info('test, %d/%d, %f', i, len(testloader), dm)


    def do_estimate(self, p0, p1):
        np_p0 = p0.cpu().contiguous().squeeze(0).numpy() # --> (N, 3)
        np_p1 = p1.cpu().contiguous().squeeze(0).numpy() # --> (M, 3)

        mod = icp.ICP(np_p0, np_p1)
        g, p, itr = mod.compute(self.max_iter)

        est_g = torch.from_numpy(g).view(-1, 4, 4).to(p0) # (1, 4, 4)

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