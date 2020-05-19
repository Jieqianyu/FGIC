import argparse
import os
import time
from utils import *
import cfg
from dataset import ListDataset
from test import validate
from logger import Logger

import torch
import torch.nn as nn
from torch.autograd import Variable
from model import SPP_VGG16

def train(train_loader, model, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (_, images, targets) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        images = Variable(images).to(args.device)
        targets = Variable(targets).to(args.device)

        # compute output
        loss, output  = model(images, targets)

        # measure accuracy and record loss
        acc1 = accuracy(output, targets)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    
    return losses.avg, top1.avg


def main(args):
    best_acc1 = 0
    os.makedirs('checkpoints', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    print('device: {}'.format(device))

    # create model
    model = SPP_VGG16().to(device)
    model.apply(weights_init_normal)
    # define optimizer
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)

            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading code
    train_path = cfg.TRAIN_PATH
    train_label_path = cfg.TRAIN_LABEL_PATH
    val_path  = cfg.VAL_PATH
    val_label_path = cfg.VAL_LABEL_PATH
    
    train_dataset = ListDataset(train_path, train_label_path)
    print(len(train_dataset))
    val_dataset = ListDataset(val_path, val_label_path)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, collate_fn=train_dataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, collate_fn=val_dataset.collate_fn)

    logger = Logger('./logs')
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        adjust_learning_rate(optimizer, epoch, args)
        train_loss, train_acc = train(train_loader, model, optimizer, epoch, args)

        # evaluate on validation set
        val_loss, val_acc = validate(val_loader, model, args)

        # remember best acc@1 and save checkpoint
        is_best = val_acc > best_acc1
        best_acc1 = max(val_acc, best_acc1)

        # log
        info = {
                'train_loss': float(train_loss),
                'train_acc': float(train_acc),
                'val_loss': float(val_loss),
                'val_acc': float(val_acc)
                }
        for tag, value in info.items():
            logger.scalar_summary(tag, value, epoch)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run (default: 90)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts, default: 0)')
    parser.add_argument('-b', '--batch-size', default=32, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate ( default: 0.01)', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')

    args = parser.parse_args()
    main(args)
