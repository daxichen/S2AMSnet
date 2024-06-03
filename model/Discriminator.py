import torch.nn.functional as F
import torch
import torch.nn as nn
from .involution import involution

class Spectral_Weight(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Spectral_Weight, self).__init__()
        self.f_inv_11 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, dilation, groups, bias)
        self.f_inv_12 = involution(in_channels, kernel_size, 1)
        self.bn_h = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X_h):
        x1 =self.f_inv_12(X_h)
        x2 =self.f_inv_11(x1)
        X_h =self.relu(self.bn_h(x2))
        return X_h
    
class Spatial_Weight(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Spatial_Weight, self).__init__()
        self.Conv_weight = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn_h = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X_h):
        X_h = self.relu(self.bn_h(self.Conv_weight(X_h)))
        return X_h

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
    
class discriminator(nn.Module):
    def __init__(self, inchannel, outchannel, num_classes, patch_size):
        super(discriminator, self).__init__()
        self.inchannel = inchannel
        self.patch_size = patch_size
        self.Weight_Alpha = nn.Parameter(torch.ones(2) / 2, requires_grad=True)
        self.Spectral_Weight_1 = Spectral_Weight(inchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.Spatial_Weight_1 = Spatial_Weight(inchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.mp = nn.MaxPool2d(2)
        self.Spectral_Weight_2 = Spectral_Weight(outchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.Spatial_Weight_2 = Spatial_Weight(outchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.Spectral_Weight_3 = Spectral_Weight(outchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.Spatial_Weight_3 = Spatial_Weight(outchannel, outchannel, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(self._get_final_flattened_size(), outchannel)
        self.relu1 = nn.ReLU(inplace=True)
        self.cls_head_src = nn.Linear(outchannel, num_classes)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, self.inchannel,
                             self.patch_size, self.patch_size))
            in_size = x.size(0)
            out1 = self.Spatial_Weight_1(x)
            out2 = self.Spatial_Weight_2(out1)
            out3 = self.mp(self.Spatial_Weight_3(out2))
            out3 = out3.view(in_size, -1)
            w, h = out3.size()
            fc_1 = w * h
        return fc_1

    def forward(self, x):
        in_size = x.size(0)
        weight_alpha1 = F.softmax(self.Weight_Alpha, dim=0)
        out1 = weight_alpha1[0] * self.Spectral_Weight_1(x) + weight_alpha1[1] * self.Spatial_Weight_1(x)

        weight_alpha2 = F.softmax(self.Weight_Alpha, dim=0)
        out2 = weight_alpha2[0] * self.Spectral_Weight_2(out1) + weight_alpha2[1] * self.Spatial_Weight_2(out1)

        weight_alpha3 = F.softmax(self.Weight_Alpha, dim=0)
        out3 = weight_alpha3[0] * self.Spectral_Weight_3(out2) + weight_alpha3[1] * self.Spatial_Weight_3(out2)
        out3 = self.mp(out3)

        out3 = out3.view(in_size, -1)
        out4 = self.relu1(self.fc1(out3))
        clss = self.cls_head_src(out4)

        return clss








