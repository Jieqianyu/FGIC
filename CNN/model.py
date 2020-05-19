import torch
import torchvision
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1)


class TwoLayerFCNet(nn.Module):
    def __init__(self, num_hidden=1024, num_class=200):
        super().__init__()
        
        self.two_fc = nn.Sequential(
            Flatten(),
            nn.Linear(3*32*32, num_hidden),
            nn.ReLU(True),
            nn.Linear(num_hidden, num_class),
        )

    def forward(self, x):
        out = self.two_fc(x)

        return out

class ConvNet(nn.Module):
    def __init__(self, num_class=200):
        super().__init__()
        layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # layer4 = nn.Sequential(
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)
        # )
        # layer5 = nn.Sequential(
        #     nn.Conv2d(128, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)
        # )
        # layer6 = nn.Sequential(
        #     nn.Conv2d(256, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)
        # )
        self.cnn = nn.Sequential(
            layer1,
            layer2,
            layer3,
            # layer4,
            # layer5,
            # layer6,
            Flatten(),
            nn.Linear(64*4*4, num_class),
        )

    def forward(self, x):
        out = self.cnn(x)

        return out

if __name__ == '__main__':
    input_test = torch.ones(4, 3, 32, 32)
    fc_net = TwoLayerFCNet()
    output_test = fc_net(input_test)
    print(output_test)
