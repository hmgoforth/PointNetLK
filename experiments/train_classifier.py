"""
    Example for training a classifier.

    It is better that before training PointNet-LK,
    train a classifier with same dataset
    so that we can use 'transfer-learning.'
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
    parser = argparse.ArgumentParser(description='PointNet classifier')

    # required.
    parser.add_argument('-o', '--outfile', required=True, type=str,
                        metavar='BASENAME', help='output filename (prefix)') # result: ${BASENAME}_feat_best.pth
    parser.add_argument('-i', '--dataset-path', required=True, type=str,
                        metavar='PATH', help='path to the input dataset') # like '/path/to/ModelNet40'
    parser.add_argument('-c', '--categoryfile', required=True, type=str,
                        metavar='PATH', help='path to the categories to be trained') # eg. './sampledata/modelnet40_half1.txt'

    # settings for input data
    parser.add_argument('--dataset-type', default='modelnet', choices=['modelnet', 'shapenet2'],
                        metavar='DATASET', help='dataset type (default: modelnet)')
    parser.add_argument('--num-points', default=1024, type=int,
                        metavar='N', help='points in point-cloud (default: 1024)')

    # settings for PointNet
    parser.add_argument('--use-tnet', dest='use_tnet', action='store_true',
                        help='flag for setting up PointNet with TNet')
    parser.add_argument('--dim-k', default=1024, type=int,
                        metavar='K', help='dim. of the feature vector (default: 1024)')
    parser.add_argument('--symfn', default='max', choices=['max', 'avg'],
                        help='symmetric function (default: max)')

    # settings for on training
    parser.add_argument('-l', '--logfile', default='', type=str,
                        metavar='LOGNAME', help='path to logfile (default: null (no logging))')
    parser.add_argument('-j', '--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--epochs', default=200, type=int,
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int,
                        metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        metavar='METHOD', help='name of an optimizer (default: Adam)')
    parser.add_argument('--resume', default='', type=str,
                        metavar='PATH', help='path to latest checkpoint (default: null (no-use))')
    parser.add_argument('--pretrained', default='', type=str,
                        metavar='PATH', help='path to pretrained model file (default: null (no-use))')
    parser.add_argument('--device', default='cuda:0', type=str,
                        metavar='DEVICE', help='use CUDA if available')

    args = parser.parse_args(argv)
    return args

def main(args):
    # dataset
    trainset, testset = get_datasets(args)
    num_classes = len(trainset.classes)

    # training
    act = Action(args, num_classes)
    run(args, trainset, testset, act)


def run(args, trainset, testset, action):
    if not torch.cuda.is_available():
        args.device = 'cpu'
    args.device = torch.device(args.device)

    LOGGER.debug('Trainer (PID=%d), %s', os.getpid(), args)

    model = action.create_model()
    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        model.load_state_dict(torch.load(args.pretrained, map_location='cpu'))
    model.to(args.device)

    checkpoint = None
    if args.resume:
        assert os.path.isfile(args.resume)
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])

    # dataloader
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    # optimizer
    min_loss = float('inf')
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(learnable_params)
    else:
        optimizer = torch.optim.SGD(learnable_params, lr=0.1)

    if checkpoint is not None:
        min_loss = checkpoint['min_loss']
        optimizer.load_state_dict(checkpoint['optimizer'])

    # training
    LOGGER.debug('train, begin')
    for epoch in range(args.start_epoch, args.epochs):
        #scheduler.step()

        running_loss, running_info = action.train_1(model, trainloader, optimizer, args.device)
        val_loss, val_info = action.eval_1(model, testloader, args.device)

        is_best = val_loss < min_loss
        min_loss = min(val_loss, min_loss)

        LOGGER.info('epoch, %04d, %f, %f, %f, %f', epoch + 1, running_loss, val_loss, running_info, val_info)
        snap = {'epoch': epoch + 1,
                'model': model.state_dict(),
                'min_loss': min_loss,
                'optimizer' : optimizer.state_dict(),}
        if is_best:
            save_checkpoint(snap, args.outfile, 'snap_best')
            save_checkpoint(model.state_dict(), args.outfile, 'model_best')
            save_checkpoint(model.features.state_dict(), args.outfile, 'feat_best')

        save_checkpoint(snap, args.outfile, 'snap_last')
        save_checkpoint(model.state_dict(), args.outfile, 'model_last')
        save_checkpoint(model.features.state_dict(), args.outfile, 'feat_last')

    LOGGER.debug('train, end')

def save_checkpoint(state, filename, suffix):
    torch.save(state, '{}_{}.pth'.format(filename, suffix))


class Action:
    def __init__(self, args, num_classes):
        self.num_classes = num_classes
        self.dim_k = args.dim_k
        self.use_tnet = args.use_tnet
        self.sym_fn = None
        if args.symfn == 'max':
            self.sym_fn = ptlk.pointnet.symfn_max
        elif args.symfn == 'avg':
            self.sym_fn = ptlk.pointnet.symfn_avg

    def create_model(self):
        feat = ptlk.pointnet.PointNet_features(self.dim_k, self.use_tnet, self.sym_fn)
        return ptlk.pointnet.PointNet_classifier(self.num_classes, feat, self.dim_k)

    def train_1(self, model, trainloader, optimizer, device):
        model.train()
        vloss = 0.0
        pred  = 0.0
        count = 0
        for i, data in enumerate(trainloader):
            target, output, loss = self.compute_loss(model, data, device)
            # forward + backward + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss1 = loss.item()
            vloss += loss1
            count += output.size(0)

            _, pred1 = output.max(dim=1)
            ag = (pred1 == target)
            am = ag.sum()
            pred += am.item()

        running_loss = float(vloss)/count
        accuracy = float(pred)/count
        return running_loss, accuracy

    def eval_1(self, model, testloader, device):
        model.eval()
        vloss = 0.0
        pred  = 0.0
        count = 0
        with torch.no_grad():
            for i, data in enumerate(testloader):
                target, output, loss = self.compute_loss(model, data, device)

                loss1 = loss.item()
                vloss += loss1
                count += output.size(0)

                _, pred1 = output.max(dim=1)
                ag = (pred1 == target)
                am = ag.sum()
                pred += am.item()

        ave_loss = float(vloss)/count
        accuracy = float(pred)/count
        return ave_loss, accuracy

    def compute_loss(self, model, data, device):
        points, target = data

        points = points.to(device)
        target = target.to(device)

        output = model(points)
        loss = model.loss(output, target)

        return target, output, loss


class ShapeNet2_transform_coordinate:
    def __init__(self):
        pass
    def __call__(self, mesh):
        return mesh.clone().rot_x()

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
                ptlk.data.transforms.Resampler(args.num_points),\
                ptlk.data.transforms.RandomRotatorZ(),\
                ptlk.data.transforms.RandomJitter()\
            ])

        trainset = ptlk.data.datasets.ModelNet(args.dataset_path, train=1, transform=transform, classinfo=cinfo)
        testset = ptlk.data.datasets.ModelNet(args.dataset_path, train=0, transform=transform, classinfo=cinfo)

    elif args.dataset_type == 'shapenet2':
        transform = torchvision.transforms.Compose([\
                ShapeNet2_transform_coordinate(),\
                ptlk.data.transforms.Mesh2Points(),\
                ptlk.data.transforms.OnUnitCube(),\
                ptlk.data.transforms.Resampler(args.num_points),\
                ptlk.data.transforms.RandomRotatorZ(),\
                ptlk.data.transforms.RandomJitter()\
            ])

        dataset = ptlk.data.datasets.ShapeNet2(args.dataset_path, transform=transform, classinfo=cinfo)
        trainset, testset = dataset.split(0.8)

    return trainset, testset


if __name__ == '__main__':
    ARGS = options()

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s:%(name)s, %(asctime)s, %(message)s',
        filename=ARGS.logfile)
    LOGGER.debug('Training (PID=%d), %s', os.getpid(), ARGS)

    main(ARGS)
    LOGGER.debug('done (PID=%d)', os.getpid())

#EOF