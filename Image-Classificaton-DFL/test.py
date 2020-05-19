import argparse
import os
import time
from utils import *
import cfg
import dataset
from model import DFL_VGG16

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda()
                target = target.cuda()

            # compute output
            output = model(images)
            out = output[0] + output[1] + 0.1 * output[2] 
            loss = criterion(output[0], target) + criterion(output[1], target) + 0.1 * criterion(output[2], target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(out, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return losses.avg, top1.avg


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: {}'.format(device))

    # create model
    print("=> creating model DFL-CNN...")
    model = DFL_VGG16(nclass=cfg.NUM_CLASSES).to(device)

    # load checkpoint
    if args.model_weight:
        if os.path.isfile(args.model_weight):
            print("=> loading checkpoint '{}'".format(args.model_weight))
            
            checkpoint = torch.load(args.model_weight, map_location=device)
            model.load_state_dict(checkpoint['state_dict'])

            print("=> loaded checkpoint '{}'".format(args.model_weight))
        else:
            print("=> no checkpoint found at '{}'".format(args.model_weight))

    # define loss function (criterion)
    criterion = nn.CrossEntropyLoss().to(device)

    # Data loading code
    testdir = cfg.VAL_DATASET_DIR
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    test_dataset = dataset.RPC_SINGLE(
        testdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    # Evaluate on test dataset
    validate(test_loader, model, criterion, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N',
                        help='mini-batch size (default: 64), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-w', '--model-weight', default=cfg.WEIGHT_DIR, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: {})'.format(cfg.WEIGHT_DIR))

    args = parser.parse_args()
    main(args)