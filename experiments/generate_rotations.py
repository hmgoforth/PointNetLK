"""
    generate perturbations for testing PointNet-LK and/or ICP
    for ex.1
"""

import argparse
import os
import sys
import logging
import math
import numpy
import torch
import torch.utils.data
import torchvision

# addpath('../')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
import ptlk

def options(argv=None):
    parser = argparse.ArgumentParser(description='PointNet-LK')

    # required.
    parser.add_argument('-o', '--outfile', required=True, type=str,
                        metavar='FILENAME', help='output filename (.csv)') # the perturbation file for 'test_pointlk.py'
    parser.add_argument('-i', '--dataset-path', required=True, type=str,
                        metavar='PATH', help='path to the input dataset') # like '/path/to/ModelNet40'
    parser.add_argument('-c', '--categoryfile', required=True, type=str,
                        metavar='PATH', help='path to the categories to be tested') # eg. './sampledata/modelnet40_half1.txt'

    # settings for input data
    parser.add_argument('--deg', default=60, type=float,
                        metavar='T', help='fixed degree of rotation (perturbations) (default: 60)')
    parser.add_argument('--max-trans', default=0.3, type=float,
                        help='max translation in each axis (default: 0.3)')
    parser.add_argument('--format', default='wv', choices=['wv', 'wt'],
                        help='output format (default: wv (twist-vector), wt means rotation- and translation-vector)')
    parser.add_argument('--dataset-type', default='modelnet', choices=['modelnet'],
                        metavar='DATASET', help='dataset type (default: modelnet)')

    args = parser.parse_args(argv)
    return args

def main(args):
    # dataset
    testset = get_datasets(args)
    batch_size = len(testset)

    amp = args.deg * math.pi / 180.0
    w = torch.randn(batch_size, 3)
    w = w / w.norm(p=2, dim=1, keepdim=True) * amp
    t = torch.rand(batch_size, 3) * args.max_trans

    if args.format == 'wv':
        # the output: twist vectors.
        R = ptlk.so3.exp(w) # (N, 3) --> (N, 3, 3)
        G = torch.zeros(batch_size, 4, 4)
        G[:, 3, 3] = 1
        G[:, 0:3, 0:3] = R
        G[:, 0:3, 3] = t

        x = ptlk.se3.log(G) # --> (N, 6)
    else:
        # rotation-vector and translation-vector
        x = torch.cat((w, t), dim=1)

    numpy.savetxt(args.outfile, x, delimiter=',')

def get_datasets(args):

    cinfo = None
    if args.categoryfile:
        #categories = numpy.loadtxt(args.categoryfile, dtype=str, delimiter="\n").tolist()
        categories = [line.rstrip('\n') for line in open(args.categoryfile)]
        categories.sort()
        c_to_idx = {categories[i]: i for i in range(len(categories))}
        cinfo = (categories, c_to_idx)

    if args.dataset_type == 'modelnet':
        transform = torchvision.transforms.Compose([\
                ptlk.data.transforms.Mesh2Points(),\
                ptlk.data.transforms.OnUnitCube(),\
            ])

        testdata = ptlk.data.datasets.ModelNet(args.dataset_path, train=0, transform=transform, classinfo=cinfo)

    return testdata


if __name__ == '__main__':
    ARGS = options()
    main(ARGS)

#EOF
