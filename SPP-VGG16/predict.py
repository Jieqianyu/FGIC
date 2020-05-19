import os
import argparse
from PIL import Image, ImageDraw, ImageFont

import torch
import torchvision

from model import SPP_VGG16
from utils import load_classes
from dataset import ListDataset
import cfg


parser = argparse.ArgumentParser(description='PyTorch RPC Predicting')
parser.add_argument('--data_path', metavar='PATH',
                    default=None, type=str, help='path to image')
parser.add_argument('--bbox_path', metavar='PATH',
                    default=None, type=str, help='path to bbox')
parser.add_argument('-c', '--classes_name', metavar='PATH',
                    default=cfg.CLASSES_NAME_PATH, type=str, help='path to classes name')                    
parser.add_argument('-w', '--model_weight', default=cfg.WEIGHTS_PATH, type=str, metavar='PATH',
                    help='path to latest checkpoint')
args = parser.parse_args()
print(args)

os.makedirs("./results", exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:{}'.format(device))

# load classes' name
classes_name = load_classes(args.classes_name)

# load data
dataset = ListDataset(args.data_path, args.bbox_path)
dataset_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, num_workers=1,
    pin_memory=True, collate_fn=dataset.collate_fn)

# create model
model = SPP_VGG16().to(device)

# load checkpoint
if args.model_weight:
    if os.path.isfile(args.model_weight):
        print("=> loading checkpoint '{}'".format(args.model_weight))
        checkpoint = torch.load(args.model_weight, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(args.model_weight))

# predict
model.eval()
for i, (path, image, targets) in enumerate(dataset_loader):
    with torch.no_grad():
        image = image.to(device)
        targets = targets.to(device)
        num_boxes, _ = targets.shape
        
        # output
        _, output = model(image, targets)
        max_values, idxes = torch.max(output, axis=1)
        
        img_org = Image.open(path[0]).convert('RGB')
        h, w = img_org.size
        
        # draw
        draw = ImageDraw.Draw(img_org)
        # font
        font = ImageFont.truetype('fonts/simsun.ttc', 48)
        # color
        color = (200, 200, 200)
        for i in range(num_boxes):
            x1 = int(targets[i, 2]*w)
            y1 = int(targets[i, 3]*h)
            w1 = int(targets[i, 4]*w)
            h1 = int(targets[i, 5]*h)
            draw.text((x1, y1-48), "{},{:.2f}".format(classes_name[idxes[i].item()], max_values[i].item()), fill=color, font=font)
            draw.rectangle([x1, y1, x1+w1, y1+h1], outline=color, width=10)
        
        print('Saving %s' % os.path.basename(path[0]).replace('jpg', 'png'))
        img_org.save(os.path.join("./results", os.path.basename(path[0]).replace('jpg', 'png')))
