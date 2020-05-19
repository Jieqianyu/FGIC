from PIL import Image
import argparse
import os
import cfg
from utils import load_classes
from model import DFL_VGG16

import torch
import torchvision.transforms as transforms
import torchvision.models as models

parser = argparse.ArgumentParser(description='PyTorch RPC Training')
parser.add_argument('-i', '--image', metavar='PATH',
                    default=cfg.IMAGE_DIR, type=str, help='path to image')
parser.add_argument('-w', '--model-weight', default=cfg.WEIGHT_DIR, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: {})'.format(cfg.WEIGHT_DIR))
args = parser.parse_args()
print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

classes_name = load_classes(cfg.CLASSES_NAME_DIR)

# Input
img = Image.open(args.image)
img = trans(img)
img = img.unsqueeze(0)
img = img.to(device)

# Create model
print("=> creating model DFL-CNN...")
model = DFL_VGG16(nclass=cfg.NUM_CLASSES)
model = model.to(device)

# load checkpoint
if args.model_weight:
    if os.path.isfile(args.model_weight):
        print("=> loading checkpoint '{}'".format(args.model_weight))

        checkpoint = torch.load(args.model_weight, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        
        print("=> loaded checkpoint '{}'".format(args.model_weight))
    else:
        print("=> no checkpoint found at '{}'".format(args.model_weight))

# Predict
model.eval()

with torch.no_grad():
    score = model(img)
    probability = torch.nn.functional.softmax(score, dim=1)
    max_value, index = torch.max(probability, 1)

    print('label:{}, true_label:{}, conf:{:.6f}'.format(
        index.item(), classes_name[index.item()], max_value.item()))
