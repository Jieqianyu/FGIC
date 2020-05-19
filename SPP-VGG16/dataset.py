import glob
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from collections import defaultdict

import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


class ListDataset(Dataset):
    def __init__(self, folder_path, annotations_path, img_size=900, mode='train'):
        self.mode = mode
        if self.mode == 'train':
            with open(annotations_path, "r") as f:
                annotations  = json.load(f)
                self.imgs_objects = {}
                for x in annotations:
                    self.imgs_objects[x["image_id"]] = x["objects"]
        elif self.mode == 'test':
            with open(annotations_path) as f:
                data = json.load(f)
                images = data['images']
                annotations = data['annotations']
                id_name = {}
                for x in images:
                    id_name[x['id']] = x['file_name']
                self.imgs_objects = defaultdict(list)
                for x in annotations:
                    self.imgs_objects[id_name[x['image_id']]].append({'bbox':x['bbox'], 'category_id':x['category_id']})
        else:
            raise ValueError('Invalid mode "%s"' % self.mode)

        self.img_files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size
        self.batch_count = 0

    def __getitem__(self, index):
        # image
        img_path = self.img_files[index % len(self.img_files)]

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert('RGB'))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape        

        #  Label
        img_objects = self.imgs_objects[os.path.basename(img_path)]

        targets = torch.zeros(len(img_objects), 6)
        for i in range(len(img_objects)):
            # Extract coordinates for unpadded + unscaled image
            x, y, w, h = img_objects[i]["bbox"]
            category_id = img_objects[i]["category_id"]
            # Adjust for added padding
            x += pad[0]
            y += pad[2]
            targets[i, 1:] = torch.tensor([category_id-1, x/padded_w, y/padded_h, w/padded_w, h/padded_h])
            
        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        for i, boxes in enumerate(targets):
             boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1

        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)


if __name__ == '__main__':
    root = r'/media/Data/dataset_jm/rpc_synthesize/val'

    dataset = ListDataset(root, r'/media/Data/dataset_jm/rpc_synthesize/synthesize_val.json', img_size = 448)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    plt.figure()
    for cnt, (path, image, targets) in enumerate(dataloader):
        print(image.shape)
        _,_, h, w = image.shape
        ax = plt.subplot(4, 4, cnt+1)
        ax.axis('off')
        
        ax.imshow(image[0].numpy().transpose(2, 1, 0))
        boxes = [target for target in targets if target[0] == 0]
        for box in boxes:
            rect = patches.Rectangle((box[3]*h, box[2]*w), box[5]*w, box[4]*h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        plt.pause(0.001)

        if cnt == 15:
            break

