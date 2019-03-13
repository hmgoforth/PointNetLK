"""
    generate perturbations for testing PointNet-LK and/or ICP
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
    parser.add_argument('--mag', default=1.0, type=float,
                        metavar='T', help='mag. of twist-vectors (perturbations) (default: 1.0)')
    parser.add_argument('--mag-randomry', dest='mag_randomly', action='store_true',
                        help='flag to no-fix mag. of twist-vectors')
    parser.add_argument('--no-translation', dest='no_translation', action='store_true',
                        help='generate twist-vectors as (w, 0) (rotation-only)')
    parser.add_argument('--dataset-type', default='modelnet', choices=['modelnet'],
                        metavar='DATASET', help='dataset type (default: modelnet)')

    args = parser.parse_args(argv)
    return args

def main(args):
    # dataset
    testset = get_datasets(args)
    batch_size = len(testset)

    if not args.no_translation:
        x = ptlk.data.datasets.CADset4tracking_fixed_perturbation.generate_perturbations(\
                batch_size, args.mag, \
                randomly=args.mag_randomly)
    else:
        x = ptlk.data.datasets.CADset4tracking_fixed_perturbation.generate_rotations(\
                batch_size, args.mag, \
                randomly=args.mag_randomly)

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
