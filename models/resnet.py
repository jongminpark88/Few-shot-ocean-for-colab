'''
import torch.nn as nn

# This ResNet network was designed following the practice of the following papers:
# TADAM: Task dependent adaptive metric for improved few-shot learning (Oreshkin et al., in NIPS 2018) and
# A Simple Neural Attentive Meta-Learner (Mishra et al., in ICLR 2018).


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        return out


class ResNet(nn.Module):

    def __init__(self, args, block=BasicBlock):
        self.inplanes = 3
        super(ResNet, self).__init__()

        self.args = args
        self.layer1 = self._make_layer(block, 64, stride=2)
        self.layer2 = self._make_layer(block, 160, stride=2)
        self.layer3 = self._make_layer(block, 320, stride=2)
        self.layer4 = self._make_layer(block, 640, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        return x
'''


'''
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class MBConvBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, expand_ratio=1):
        super(MBConvBlock, self).__init__()
        self.stride = stride

        hidden_dim = int(inplanes * expand_ratio)
        self.use_res_connect = self.stride == 1 and inplanes == planes

        layers = []
        if expand_ratio != 1:
            # Point-wise expansion
            layers.append(conv1x1(inplanes, hidden_dim))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.SiLU())

        # Depthwise convolution
        layers.append(conv3x3(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.SiLU())

        # Point-wise projection
        layers.append(conv1x1(hidden_dim, planes))
        layers.append(nn.BatchNorm2d(planes))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

class ResNet(nn.Module):
    def __init__(self, args, block=MBConvBlock):
        super(ResNet, self).__init__()
        self.inplanes = 3
        self.args = args

        self.layer1 = self._make_layer(block, 64, stride=2, expand_ratio=1)
        self.layer2 = self._make_layer(block, 160, stride=2, expand_ratio=6)
        self.layer3 = self._make_layer(block, 320, stride=2, expand_ratio=6)
        self.layer4 = self._make_layer(block, 640, stride=2, expand_ratio=6)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride, expand_ratio):
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, expand_ratio=expand_ratio))
        self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
'''




import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer


class ResNet(nn.Module):
    def __init__(self, args, pretrained=True):
        super(ResNet, self).__init__()
        self.args = args

        # Load pretrained Swin Transformer with adjusted input size
        self.swin = SwinTransformer(pretrained=pretrained, img_size=84, in_chans=3, num_classes=0)

        # Custom layer to map Swin Transformer output to desired shape
        self.feature_adapter = nn.Sequential(
            nn.Linear(self.swin.num_features, 640 * 6 * 6),  # Map to required size
            nn.GELU()
        )

    def forward(self, x):
        if len(x.shape) != 4:
            raise ValueError("Input tensor must have 4 dimensions (batch, channels, height, width).")

        b, c, h, w = x.shape

        if h != 84 or w != 84:
            raise ValueError(f"Input height and width must be 84x84, but got {h}x{w}.")

        x = self.swin(x)  # Feature extraction using Swin Transformer

        # Adjust Swin Transformer output to match desired 4D shape
        x = self.feature_adapter(x)  # Output shape: (batch, 640*6*6)
        x = x.view(b, 640, 6, 6)  # Reshape to 4D tensor

        return x





'''
import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)

        return out


class ResNet(nn.Module):
    def __init__(self, args, pretrained=True):
        super(ResNet, self).__init__()
        self.args = args

        # Load pretrained Swin Transformer with adjusted input size
        self.swin = SwinTransformer(pretrained=pretrained, img_size=84, in_chans=3, num_classes=0)

        # Custom layer to map Swin Transformer output to desired shape
        self.feature_adapter = nn.Sequential(
            nn.Linear(self.swin.num_features, 64 * 6 * 6),  # Map to required size
            nn.GELU()
        )
        self.layer1 = self._make_layer(block, 64, stride=2)
        self.layer2 = self._make_layer(block, 160, stride=2)
        self.layer3 = self._make_layer(block, 320, stride=2)
        self.layer4 = self._make_layer(block, 640, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)


    def forward(self, x):
        if len(x.shape) != 4:
            raise ValueError("Input tensor must have 4 dimensions (batch, channels, height, width).")

        b, c, h, w = x.shape

        if h != 84 or w != 84:
            raise ValueError(f"Input height and width must be 84x84, but got {h}x{w}.")

        x = self.swin(x)  # Feature extraction using Swin Transformer

        # Adjust Swin Transformer output to match desired 4D shape
        x = self.feature_adapter(x)  # Output shape: (batch, 640*6*6)
        x = x.view(b, 640, 6, 6)  # Reshape to 4D tensor

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

'''


