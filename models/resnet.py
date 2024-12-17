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



#upgrade resnet
'''
import torch
import torch.nn as nn
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(0.3)
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

        # Ensure the residual and out sizes match
        if residual.size() != out.size():
            padding = (0, out.size(3) - residual.size(3), 0, out.size(2) - residual.size(2))
            residual = nn.functional.pad(residual, padding)

        out += residual
        out = self.relu(out)
        out = self.dropout(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
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

        # Ensure the residual and out sizes match
        if residual.size() != out.size():
            padding = (0, out.size(3) - residual.size(3), 0, out.size(2) - residual.size(2))
            residual = nn.functional.pad(residual, padding)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, args, block=BasicBlock):
        self.inplanes = 3
        super(ResNet, self).__init__()

        self.args = args
        self.layer1 = self._make_layer(block, 64, stride=2)  # Output: (48, 64, 42, 42)
        self.layer2 = self._make_layer(block, 160, stride=2) # Output: (48, 160, 21, 21)
        self.layer3 = self._make_layer(block, 320, stride=2) # Output: (48, 320, 11, 11)
        self.layer4 = self._make_layer(block, 640, stride=1) # Output: (48, 640, 6, 6)

        # Squeeze-and-Excitation block
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(640, 640 // 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(640 // 16, 640, kernel_size=1),
            nn.Sigmoid()
        )

    def _make_layer(self, block, planes, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
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

        # Apply Squeeze-and-Excitation
        se_weight = self.se(x)
        x = x * se_weight

        return x
'''




#efficientnet
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


#단순화한 eff

import torch
import torch.nn as nn
import torch.nn.functional as F
# Depthwise Separable Convolution Block
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.conv = nn.Sequential(
            # Depthwise Convolution
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            # Pointwise Convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU()
        )

    def forward(self, x):
        return self.conv(x)

# Simplified EfficientNet-inspired Block
class SimplifiedMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=4, use_se=False):
        super(SimplifiedMBConv, self).__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        ) if expand_ratio > 1 else nn.Identity()

        self.depthwise_separable = DepthwiseSeparableConv(hidden_dim, out_channels, stride)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1),
            nn.Sigmoid()
        ) if use_se else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.expand(x)
        x = self.depthwise_separable(x)
        if isinstance(self.se, nn.Sequential):  # If SE is enabled
            x = x * self.se(x)
        if self.use_residual:
            return x + identity
        return x

# Simplified EfficientNet Model
class ResNet(nn.Module):
    def __init__(self, args):
        super(ResNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )

        self.args=args

        self.blocks = nn.Sequential(
            SimplifiedMBConv(32, 64, stride=1, expand_ratio=1),  # Stage 1
            SimplifiedMBConv(64, 128, stride=2),                # Stage 2
            SimplifiedMBConv(128, 256, stride=2),               # Stage 3
            SimplifiedMBConv(256, 512, stride=2),               # Stage 4
            SimplifiedMBConv(512, 640, stride=2)                # Stage 5
        )

        # Final resizing to (6, 6)
        self.final_resize = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.final_resize(x)  # Resize to (Batch, 640, 6, 6)
        return x






#effi_b7
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b7

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        # Load EfficientNet-B7 model structure
        self.model = efficientnet_b7(pretrained=False)

        # Modify the stem to accept input size (48, 3, 84, 84)
        self.model.features[0][0] = nn.Conv2d(3, self.model.features[0][0].out_channels,
                                              kernel_size=3, stride=1, padding=1, bias=False)

        # Modify the final layers
        self.model.classifier = nn.Identity()  # Remove classification head
        self.model.avgpool = nn.Identity()  # Remove global average pooling

        # Add a Conv2d layer to reduce channels to 640
        self.channel_reduction = nn.Conv2d(2560, 640, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        x = self.model.features(x)  # Feature extraction
        x = self.channel_reduction(x)  # Reduce channels to 640
        x = F.interpolate(x, size=(6, 6), mode='bilinear', align_corners=False)  # Resize to (6, 6)
        return x
'''


#clip
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

class ResNet(nn.Module):
    def __init__(self,args):
        super(ResNet, self).__init__()
        # Load pre-trained CLIP model (ViT-B/32 as the backbone)
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32', pretrained='openai')
        self.encoder = model.visual  # Use the visual encoder of CLIP
        self.args=args
        # Add a Conv2d layer to adjust the channel dimension to 640
        self.channel_adjust = nn.Conv2d(512, 640, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        # Resize input to match CLIP's expected input size
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # Pass through CLIP encoder
        x = self.encoder(x)  # Output shape: (Batch, Features)

        # Dynamically calculate grid size
        batch_size, total_features = x.shape
        grid_size = int((total_features // 512) ** 0.5)

        if grid_size * grid_size * 512 != total_features:
            raise ValueError(f"Unexpected output size: {x.shape}")

        # Reshape into (Batch, Channels, Height, Width)
        x = x.view(batch_size, 512, grid_size, grid_size)

        # Adjust channels to 640
        x = self.channel_adjust(x)

        # Resize output to (6, 6)
        x = F.interpolate(x, size=(6, 6), mode='bilinear', align_corners=False)
        return x
'''



#swin
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





#swin+resnet34
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


