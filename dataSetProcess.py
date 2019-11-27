import torch.utils.data
import torch
from torchvision import transforms
import torchvision
import json
from PIL import Image

transform_1 = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
   ]
)

transform_2 = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
   ]
)

class InstantNoodles(torch.utils.data.Dataset):
   def __init__(self, json_root, transform=transform_1):
      super().__init__()
      with open(json_root, 'r') as f:
          self.images_info = json.load(f)
      self.transform = transform
   def __getitem__(self, idx):
      image_info = self.images_info[idx]
      path = image_info['path']
      tag = image_info['tag']
      bbox = image_info['bbox']
      # read image
      image = Image.open(path)
      # extract ROI
      image = image.crop(tuple(bbox))
      # transform
      image = self.transform(image)

      return image, tag
   def __len__(self):
      return len(self.images_info)

def Birds(root):
   return torchvision.datasets.ImageFolder(root, transform_2)

if __name__ == '__main__':
   #train_set = InstantNoodles(r'./instant_noodles/train.json')
   train_set = Birds(r'./birds/train')
   train_loader = torch.utils.data.DataLoader(train_set, batch_size = 4, shuffle=True)
   print(type(train_loader))
   data_iter = iter(train_loader)
   images, lables = data_iter.next()
   print(type(images), type(lables))
   print(images.size(), lables.size())
