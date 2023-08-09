'''
AI Fashion Coordinator
(Baseline For Fashion-How Challenge)

MIT License

Copyright (C) 2022, Integrated Intelligence Research Section, ETRI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Update: 2022.04.20.
'''
import torch.nn as nn
import torchvision.models as models

from networks import register

class ResExtractor(nn.Module):
    """Feature extractor based on ResNet structure
        Selectable from resnet18 to resnet152

    Args:
        resnetnum: Desired resnet version
                    (choices=['18','34','50','101','152'])
        pretrained: 'True' if you want to use the pretrained weights provided by Pytorch,
                    'False' if you want to train from scratch.
    """

    def __init__(self, resnetnum='50', pretrained=True):
        super(ResExtractor, self).__init__()

        if resnetnum == '18':
            self.resnet = models.resnet18(pretrained=pretrained) # 512
        elif resnetnum == '34':
            self.resnet = models.resnet34(pretrained=pretrained) # 512
        elif resnetnum == '50':     
            self.resnet = models.resnet50(pretrained=pretrained) # 2048
        elif resnetnum == '101':
            self.resnet = models.resnet101(pretrained=pretrained) # 2048
        elif resnetnum == '152':
            self.resnet = models.resnet152(pretrained=pretrained) # 2048

        self.modules_front = list(self.resnet.children())[:-2]
        self.model_front = nn.Sequential(*self.modules_front)

    def front(self, x):
        """ In the resnet structure, input 'x' passes through conv layers except for fc layers. """
        return self.model_front(x)

@register('Baseline_ResNet_emo')
class Baseline_ResNet_emo(nn.Module):
    """ Classification network of emotion categories based on ResNet18 structure. """
    
    def __init__(self, resnetnum, input_dim, hidden_dim):
        super(Baseline_ResNet_emo, self).__init__()
        resnetnum = str(resnetnum)
        self.encoder = ResExtractor(resnetnum)
        self.avg_pool = nn.AvgPool2d(kernel_size=7)

        # self.classifier1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 7))
        # self.classifier2 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 6))
        # self.classifier3 = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 3))

        self.classifier1 = nn.Linear(input_dim, 6)
        self.classifier2 = nn.Linear(input_dim, 5)
        self.classifier3 = nn.Linear(input_dim, 3)

    def forward(self, x):
        """ Forward propagation with input 'x' """
        feat = self.encoder.front(x)
        feat = self.avg_pool(feat).squeeze()
        
        y1 = self.classifier1(feat)
        y2 = self.classifier2(feat)
        y3 = self.classifier3(feat)

        return y1, y2, y3


if __name__ == '__main__':
    pass
