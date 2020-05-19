import shutil
import torch
import torch.nn as nn
import cfg

def save_checkpoint(state, is_best, filename='checkpoints/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'checkpoints/model_best.pth.tar')


def load_classes(path):
    classes_name = []
    with open(path, 'r') as f:
      classes_name = [name.replace('\n', '') for name in f]
    return classes_name


def weights_init_normal(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset() 

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, targets):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        pred = torch.argmax(output, axis=1, keepdim=False)
        label = targets[:, 1].long()
        correct = pred.eq(label)
       
        res = correct.float().sum()/output.shape[0]*100

        return res


if __name__ == "__main__":
    output = torch.tensor([[1, 2, 3], [2, 3, 4]])
    targets = torch.tensor([[0, 2, 3], [1, 2, 3]])
    print(accuracy(output, targets))

