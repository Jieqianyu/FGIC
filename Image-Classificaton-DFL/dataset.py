import torch
from PIL import Image
import matplotlib.pyplot as plt
import os

class RPC_SINGLE(torch.utils.data.Dataset):
   def __init__(self, path, transform=None):
      super().__init__()

      self.path = path
      self.transform = transform
      with open(path, 'r') as f:
          self.imgs_info = f.readlines()

   def __getitem__(self, idx):
      img_info = self.imgs_info[idx].replace('\n', '')
      elem = img_info.split(' ')
      # img_path = os.path.join(os.path.dirname(self.path), elem[0])
      img_path = os.path.join(os.path.dirname(self.path), elem[0]).replace('images_cropped', 'images_raw')
      label = int(elem[1])

      # read image
      image = Image.open(img_path)
      # transform
      if self.transform:
         image = self.transform(image)

      return image, label

   def __len__(self):
      return len(self.imgs_info)

if __name__ == '__main__':
   root = r'/media/Data/dataset_jm/single-object-rpc'

   train_dataset = RPC_SINGLE(os.path.join(root, 'train.txt'))
   test_dataset = RPC_SINGLE(os.path.join(root, 'test.txt'))
   trainval_dataset = RPC_SINGLE(os.path.join(root, 'val.txt'))

   train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 1, shuffle=True)
   test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle=True)
   trainval_loader = torch.utils.data.DataLoader(trainval_dataset, batch_size = 1, shuffle=False)

   print(type(train_loader), len(train_loader), len(test_loader), len(trainval_loader))

   with open(os.path.join(root, 'class.names'), 'r') as f:
      classes_name = [name.replace('\n', '') for name in f]

   print(classes_name)

   plt.figure()
   for cnt, (image, label) in enumerate(train_dataset):
     ax = plt.subplot(4, 4, cnt+1)
     ax.axis('off')
     ax.imshow(image)
     ax.set_title('label:{} {}'.format(label, classes_name[label]))
     plt.pause(0.001)

     if cnt == 15:
        break
