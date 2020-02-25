from torch import nn
from torch.nn import functional as F
from .BasicNet import BasicModule


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out, True)


class ResNet34(BasicModule):
    def __init__(self, num_class=2):
        super(ResNet34, self).__init__()
        self.model_name = "ResNet34"

        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2, 1)
        )

        self.seq = nn.Sequential(
            self.make_layer(64, 128, 3),
            self.make_layer(128, 256, 4, stride=2),
            self.make_layer(256, 512, 6, stride=2),
            self.make_layer(512, 512, 3, stride=2)
        )

        self.fc = nn.Linear(512, num_class)

    def make_layer(self, inchannel, outchannel, block_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.BatchNorm2d(outchannel)
        )

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1,block_num):
            layers.append(ResidualBlock(outchannel,outchannel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.seq(x)
        x = F.avg_pool2d(x,7)
        x = x.view(x.size(0), -1)
        return self.fc(x)
