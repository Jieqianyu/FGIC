import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable


class SPPLayer(torch.nn.Module):
    def __init__(self, out_pool_size=[1, 2, 4], pool_type='max_pool'):
        super().__init__()

        self.out_pool_size = out_pool_size
        self.pool_type = pool_type

    def forward(self, x):
        num, c, h, w = x.size()
        for i in range(len(self.out_pool_size)):
            kh = math.floor(h / self.out_pool_size[i]) + h%self.out_pool_size[i]
            kw = math.floor(w / self.out_pool_size[i]) + w%self.out_pool_size[i]
            sh = math.floor(h / self.out_pool_size[i])
            sw = math.floor(w / self.out_pool_size[i])

            # method of pooling 
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=(kh, kw), stride=(sh, sw)).view(num, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=(kh, kw), stride=(sh, sw)).view(num, -1)

            # flatten, concatenate
            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
                
        return x_flatten


class SPP_VGG16(nn.Module):
    def __init__(self, nclass = 200):
        super().__init__()
            
        vgg16featuremap = torchvision.models.vgg16_bn(pretrained=True).features
        self.conv = torch.nn.Sequential(*list(vgg16featuremap.children())[:23])
        self.spp_layer = SPPLayer(out_pool_size=[1, 2, 4])
        self.fc = nn.Linear(256*21, nclass)
        self.softmax = nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, targets):
        features = self.conv(x)
        batch_size,  _, h, w = features.shape
        
        output = []
        for i in range(batch_size):
            boxes = torch.stack([target for target in targets if target[0]==i])
            num, _ = boxes.shape
            # x, y, w, h
            boxes[:, 2] *= w
            boxes[:, 3] *= h
            boxes[:, 4] *= w
            boxes[:, 5] *= h
            boxes =  boxes.long()
            boxes[boxes[:, 4] < 4, 4] = 4
            boxes[boxes[:, 5] < 4, 5] = 4

            objects_features = [self.spp_layer(features[i, :, boxes[j,3]:boxes[j,3]+boxes[j,5], boxes[j,2]:boxes[j,2]+boxes[j,4]].unsqueeze(0)) for j in range(num)]
            objects_features  = torch.cat(objects_features, 0)

            out = self.fc(objects_features)
            out = self.softmax(out)
            if i == 0:
                loss = self.loss(out, boxes[:, 1])
            else:
                loss += self.loss(out, boxes[:, 1])
            output.append(out)

        loss /= batch_size
        output = torch.cat(output)

        return loss, output
        

if __name__ == '__main__':
    input_test = torch.ones(2, 3, 900, 900)
    targets = torch.tensor([[0, 1, 0.2, 0.2, 0.2, 0.2], [0, 2, 0.1, 0.1, 0.1, 0.1], [1, 3, 0.1, 0.1, 0.1, 0.1], [1, 4, 0.2, 0.2, 0.2, 0.2]])
    net = SPP_VGG16()
    loss, _  = net(input_test, targets)
    print(loss)	
        
  
