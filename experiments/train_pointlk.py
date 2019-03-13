"""
    Example for training a tracker (PointNet-LK).

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
                        metavar='BASENAME', help='output filename (prefix)') # the result: ${BASENAME}_model_best.pth
    parser.add_argument('-i', '--dataset-path', required=True, type=str,
                        metavar='PATH', help='path to the input dataset') # like '/path/to/ModelNet40'
    parser.add_argument('-c', '--categoryfile', required=True, type=str,
                        metavar='PATH', help='path to the categories to be trained') # eg. './sampledata/modelnet40_half1.txt'

    # settings for input data
    parser.add_argument('--dataset-type', default='modelnet', choices=['modelnet', 'shapenet2'],
                        metavar='DATASET', help='dataset type (default: modelnet)')
    parser.add_argument('--num-points', default=1024, type=int,
                        metavar='N', help='points in point-cloud (default: 1024)')
    parser.add_argument('--mag', default=0.8, type=float,
                        metavar='T', help='max. mag. of twist-vectors (perturbations) on training (default: 0.8)')

    # settings for PointNet
    parser.add_argument('--pointnet', default='tune', type=str, choices=['fixed', 'tune'],
                        help='train pointnet (default: tune)')
    parser.add_argument('--transfer-from', default='', type=str,
                        metavar='PATH', help='path to pointnet features file')
    parser.add_argument('--dim-k', default=1024, type=int,
                        metavar='K', help='dim. of the feature vector (default: 1024)')
    parser.add_argument('--symfn', default='max', choices=['max', 'avg'],
                        help='symmetric function (default: max)')

    # settings for LK
    parser.add_argument('--max-iter', default=10, type=int,
                        metavar='N', help='max-iter on LK. (default: 10)')
    parser.add_argument('--delta', default=1.0e-2, type=float,
                        metavar='D', help='step size for approx. Jacobian (default: 1.0e-2)')
    parser.add_argument('--learn-delta', dest='learn_delta', action='store_true',
                        help='flag for training step size delta')

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

    # training
    act = Action(args)
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

        save_checkpoint(snap, args.outfile, 'snap_last')
        save_checkpoint(model.state_dict(), args.outfile, 'model_last')

    LOGGER.debug('train, end')

def save_checkpoint(state, filename, suffix):
    torch.save(state, '{}_{}.pth'.format(filename, suffix))


class Action:
    def __init__(self, args):
        # PointNet
        self.pointnet = args.pointnet # tune or fixed
        self.transfer_from = args.transfer_from
        self.dim_k = args.dim_k
        self.sym_fn = None
        if args.symfn == 'max':
            self.sym_fn = ptlk.pointnet.symfn_max
        elif args.symfn == 'avg':
            self.sym_fn = ptlk.pointnet.symfn_avg
        # LK
        self.delta = args.delta
        self.learn_delta = args.learn_delta
        self.max_iter = args.max_iter
        self.xtol = 1.0e-7
        self.p0_zero_mean = True
        self.p1_zero_mean = True

        self._loss_type = 1 # see. self.compute_loss()

    def create_model(self):
        ptnet = self.create_pointnet_features()
        return self.create_from_pointnet_features(ptnet)

    def create_pointnet_features(self):
        ptnet = ptlk.pointnet.PointNet_features(self.dim_k, use_tnet=False, sym_fn=self.sym_fn)
        if self.transfer_from and os.path.isfile(self.transfer_from):
            ptnet.load_state_dict(torch.load(self.transfer_from, map_location='cpu'))
        if self.pointnet == 'tune':
            pass
        elif self.pointnet == 'fixed':
            for param in ptnet.parameters():
                param.requires_grad_(False)
        return ptnet

    def create_from_pointnet_features(self, ptnet):
        return ptlk.pointlk.PointLK(ptnet, self.delta, self.learn_delta)

    def train_1(self, model, trainloader, optimizer, device):
        model.train()
        vloss = 0.0
        gloss = 0.0
        count = 0
        for i, data in enumerate(trainloader):
            loss, loss_g = self.compute_loss(model, data, device)

            # forward + backward + optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            vloss1 = loss.item()
            vloss += vloss1
            gloss1 = loss_g.item()
            gloss += gloss1
            count += 1

        ave_vloss = float(vloss)/count
        ave_gloss = float(gloss)/count
        return ave_vloss, ave_gloss

    def eval_1(self, model, testloader, device):
        model.eval()
        vloss = 0.0
        gloss = 0.0
        count = 0
        with torch.no_grad():
            for i, data in enumerate(testloader):
                loss, loss_g = self.compute_loss(model, data, device)

                vloss1 = loss.item()
                vloss += vloss1
                gloss1 = loss_g.item()
                gloss += gloss1
                count += 1

        ave_vloss = float(vloss)/count
        ave_gloss = float(gloss)/count
        return ave_vloss, ave_gloss

    def compute_loss(self, model, data, device):
        p0, p1, igt = data
        p0 = p0.to(device) # template
        p1 = p1.to(device) # source
        igt = igt.to(device) # igt: p0 -> p1
        r = ptlk.pointlk.PointLK.do_forward(model, p0, p1, self.max_iter, self.xtol,\
                                            self.p0_zero_mean, self.p1_zero_mean)
        #r = model(p0, p1, self.max_iter)
        est_g = model.g

        loss_g = ptlk.pointlk.PointLK.comp(est_g, igt)

        if self._loss_type == 0:
            loss_r = ptlk.pointlk.PointLK.rsq(r)
            loss = loss_r
        elif self._loss_type == 1:
            loss_r = ptlk.pointlk.PointLK.rsq(r)
            loss = loss_r + loss_g
        elif self._loss_type == 2:
            pr = model.prev_r
            if pr is not None:
                loss_r = ptlk.pointlk.PointLK.rsq(r - pr)
            else:
                loss_r = ptlk.pointlk.PointLK.rsq(r)
            loss = loss_r + loss_g
        else:
            loss = loss_g

        return loss, loss_g


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
            ])

        traindata = ptlk.data.datasets.ModelNet(args.dataset_path, train=1, transform=transform, classinfo=cinfo)
        testdata = ptlk.data.datasets.ModelNet(args.dataset_path, train=0, transform=transform, classinfo=cinfo)

        mag_randomly = True
        trainset = ptlk.data.datasets.CADset4tracking(traindata,\
                        ptlk.data.transforms.RandomTransformSE3(args.mag, mag_randomly))
        testset = ptlk.data.datasets.CADset4tracking(testdata,\
                        ptlk.data.transforms.RandomTransformSE3(args.mag, mag_randomly))

    elif args.dataset_type == 'shapenet2':
        transform = torchvision.transforms.Compose([\
                ShapeNet2_transform_coordinate(),\
                ptlk.data.transforms.Mesh2Points(),\
                ptlk.data.transforms.OnUnitCube(),\
                ptlk.data.transforms.Resampler(args.num_points),\
            ])

        dataset = ptlk.data.datasets.ShapeNet2(args.dataset_path, transform=transform, classinfo=cinfo)
        traindata, testdata = dataset.split(0.8)

        mag_randomly = True
        trainset = ptlk.data.datasets.CADset4tracking(traindata,\
                        ptlk.data.transforms.RandomTransformSE3(args.mag, mag_randomly))
        testset = ptlk.data.datasets.CADset4tracking(testdata,\
                        ptlk.data.transforms.RandomTransformSE3(args.mag, mag_randomly))


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