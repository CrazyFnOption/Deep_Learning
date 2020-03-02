from torch import nn
from torchvision.models import vgg16
from collections import namedtuple

class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16,self).__init__()
        feature = list(vgg16(pretrained=True).features)[:23]
        self.features = nn.ModuleList(feature).eval()


    def forward(self, x):
        result = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in [3, 8, 15, 22]:
                result.append(x)
        Resultset = namedtuple("VggOutputs",['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        return Resultset(*result)