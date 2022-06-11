import torch
import torch.nn as  nn
import torch.nn.functional as func
 


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, n_in_channels, n_out_channels, stride=1):
        super(BasicBlock,self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(n_in_channels, n_out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(n_out_channels)

        self.conv2 = nn.Conv2d(n_out_channels, n_out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(n_out_channels)

        self.relu = nn.ReLU()

        self.identity_map = nn.Sequential()
        if stride == 2:
           self.identity_map = nn.Sequential(
                                        nn.Conv2d(num_in, num_out, kernel_size=1, stride=2),
                                        nn.BatchNorm2d(num_out))


    def forward(self, x):
      identity = self.identity_map(x)

      x = self.bn1(self.conv1(x))
      x = self.relu(x)

      x = self.bn2(self.conv2(x))
      x += identity
      x = self.relu(x)

      return x

class ResNet(nn.Module):
    def __init__(self, layer_list, num_classes, num_channels=3):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(num_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)

        self.layer1 = self._make_layer(BasicBlock, layer_list[0], planes=self.in_channels)
        self.layer2 = self._make_layer(BasicBlock, layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_layer(BasicBlock, layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_layer(BasicBlock, layer_list[3], planes=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*BasicBlock.expansion, num_classes)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, Block, n_blocks, planes, stride=1):
        layers = []

        layers.append(Block(self.in_channels, planes, stride=stride))
        self.in_channels = planes*Block.expansion

        for i in range(n_blocks-1):
            layers.append(Block(self.in_channels, planes))

        return nn.Sequential(*layers)

def ResNet50(num_classes, channels=3):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)