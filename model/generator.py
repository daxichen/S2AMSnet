import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Function
from torch.nn import init
from .involution import involution

class Spectral_Weight(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Spectral_Weight, self).__init__()
        self.f_inv_11 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, dilation, groups, bias)
        self.f_inv_12 = involution(in_channels, kernel_size, 1)
        self.bn_h = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X_h):
        X_h = self.f_inv_11(self.f_inv_12(X_h))
        return X_h
    
class Spatial_Weight(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, groups=1, bias=False):
        super(Spatial_Weight, self).__init__()
        self.Conv_weight = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.bn_h = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, X_h):
        X_h = self.Conv_weight(X_h)
        return X_h

class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)
        self.style_dim = style_dim

    def forward(self, x):
        s = torch.randn(x.shape[0],self.style_dim).cuda()
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        return (1 + gamma) * self.norm(x) + beta

class NormalNorm:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        size = weight.size()
        weight_mat = weight.contiguous().view(size[0], -1)
        with torch.no_grad():
            mu = weight_mat.mean()
            std = weight_mat.std()
        weight_sn = (weight-mu) / std

        return weight_sn

    @staticmethod
    def apply(module, name):
        fn = NormalNorm(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', weight)
        module.register_buffer(name, weight)

        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight_sn = self.compute_weight(module)
        setattr(module, self.name, weight_sn)


def spectral_norm(module, name='weight'):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        NormalNorm.apply(module, name)

    return module


def spectral_init(module, gain=1):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        init.xavier_uniform_(module.weight, gain)

    return spectral_norm(module)


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


class SpectralNet(nn.Module):
    def __init__(self, args):
        super(SpectralNet, self).__init__()
        ch = args.GIN_ch
        self.Spectral_Weight_11 = Spectral_Weight(args.n_bands, ch, kernel_size=3, stride=1, padding=1)
        self.AdaIN1 = AdaIN(2,ch) if args.noise else nn.Identity()
        self.Spectral_Weight_12 = Spectral_Weight(ch, ch, kernel_size=3, stride=1, padding=1)
        self.Spectral_Weight_13 = Spectral_Weight(ch, ch, kernel_size=3, stride=1, padding=1)
        self.generate1 = nn.Conv2d(ch, args.n_bands, 3, padding=1)
        self.activate1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(ch)
        self.Weight_Alpha1 = nn.Parameter(torch.ones(2) / 2, requires_grad=True)
        self.__initialize_weights()

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 1.0)

    def normalize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):

                weight = m.weight.data*1.0
                m.weight.data = (m.weight.data - weight.mean())/weight.std()

    def forward(self, x):
        out1 = self.Spectral_Weight_11(x)
        out1 = self.activate1(self.bn1(self.AdaIN1(out1)))
        out1 = self.Spectral_Weight_12(out1)
        out1 = self.activate1(self.bn1(self.AdaIN1(out1)))
        out1 = self.Spectral_Weight_13(out1)
        out1 = self.activate1(self.bn1(self.AdaIN1(out1)))
        out1 = self.generate1(out1)
        weight_alpha1 = F.softmax(self.Weight_Alpha1, dim=0)
        out1 = weight_alpha1[0] * x + weight_alpha1[1] * out1
        return out1

class SpatialNet(nn.Module):
    def __init__(self, args):
        super(SpatialNet, self).__init__()
        ch = args.GIN_ch
        self.Spatial_Weight_21 = Spatial_Weight(args.n_bands, ch, kernel_size=3, stride=1, padding=1)
        self.AdaIN2 = AdaIN(2, ch) if args.noise else nn.Identity()
        self.Spatial_Weight_22 = Spatial_Weight(ch, ch, kernel_size=3, stride=1, padding=1)
        self.Spatial_Weight_23 = Spatial_Weight(ch, ch, kernel_size=3, stride=1, padding=1)
        self.generate2 = nn.Conv2d(ch, args.n_bands, 3, padding=1)
        self.activate2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(ch)
        self.Weight_Alpha2 = nn.Parameter(torch.ones(2) / 2, requires_grad=True)
        self.__initialize_weights()
       
    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 1.0)

    def normalize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):

                weight = m.weight.data*1.0
                m.weight.data = (m.weight.data - weight.mean())/weight.std()

    def forward(self, x):
        out2 = self.Spatial_Weight_21(x)
        out2 = self.activate2(self.bn2(self.AdaIN2(out2)))
        out2 = self.Spatial_Weight_22(out2)
        out2 = self.activate2(self.bn2(self.AdaIN2(out2)))
        out2 = self.Spatial_Weight_23(out2)
        out2 = self.activate2(self.bn2(self.AdaIN2(out2)))
        out2 = self.generate2(out2)
        weight_alpha2 = F.softmax(self.Weight_Alpha2, dim=0)
        out2 = weight_alpha2[0] * x + weight_alpha2[1] * out2

        return out2

class SSDGnet(nn.Module):
    def __init__(self, args):
        super(SSDGnet, self).__init__()

        self.Net1 = SpectralNet(args)
        self.Net2 = SpatialNet(args)

    def __initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 1.0)

    def normalize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                weight = m.weight.data*1.0
                m.weight.data = (m.weight.data - weight.mean())/weight.std()

    def forward(self, x):

        out1 = self.Net1(x)
        out2 = self.Net2(x)

        return out1, out2
