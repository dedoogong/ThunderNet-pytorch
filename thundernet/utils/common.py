import torch
import torch.nn as nn
import torchvision
from torch.functional import F

class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels, strides=1, groups=1):
        super(Conv1x1, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=strides, groups=groups)

    def forward(self, x):
        out = self.conv1x1(x)
        return out

class Conv3x3(nn.Module):
    def __init__(self, in_channels, out_channels, strides=1, padding=1, groups=1):
        super(Conv3x3, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, stride=strides, groups=groups)

    def forward(self, x):
        out = self.conv3x3(x)
        return out

class DepthwiseConv5x5(nn.Module):
    def __init__(self, channels, stride):
        super(DepthwiseConv5x5, self).__init__()
        self.depthwise_conv5x5 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=5, stride=stride, padding=2, groups=channels)

    def forward(self, x):
        out = self.depthwise_conv5x5(x)
        return out

class ConvBlock(nn.Module):
    def __init__(self,in_channels, out_channels, kernel_size, strides, padding, dilation=1, groups=1,
               use_bias=False, activation="relu", activate=True):
        super(ConvBlock).__init__()
        self.activate = activate
        self.activation=activation
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=strides, padding=padding, dilation=dilation, groups=groups)
        self.batchnorm=nn.BatchNorm2d(momentum=0.9, eps=1e-5)

    def forward(self,x):
        x = self.conv(x)
        x = self.batchnorm(x)
        if self.activate:
            assert (self.activation is not None)
            #if isfunction(self.activation):
            #    x = self.activation(x)
            if isinstance(self.activation, str):
                if self.activation == "relu":
                    x = F.relu(x)
                elif self.activation == "relu6":
                    x = F.relu6(x)
                else:
                    raise NotImplementedError()
            #else:
            #    x = self.activation(x)
        return x

class Conv1x1Block(nn.Module):
    def __init__(self, in_channels,out_channels,strides=1,groups=1,use_bias=False,
                  activation="relu",activate=True,name="conv1x1_block"):
        super(Conv1x1Block).__init__()
        self.conv=ConvBlock(in_channels=in_channels,out_channels=out_channels,
            kernel_size=1,strides=strides,padding=0, groups=groups,
            use_bias=use_bias,activation=activation,activate=activate,name=name)

    def forward(self, x):
        self.conv(x)
        return x

class Conv3x3Block(nn.Module):

    def __init__(self, in_channels,out_channels,strides=1,padding=1,dilation=1,
                  groups=1,use_bias=False,activation="relu",activate=True,name="conv3x3_block"):
        super(Conv3x3Block).__init__()
        self.conv=ConvBlock(in_channels=in_channels,out_channels=out_channels,kernel_size=3,strides=strides,
                            padding=padding,dilation=dilation,groups=groups,
                            use_bias=use_bias,activation=activation,activate=activate,name=name)

    def forward(self, x):
        return self.conv(x)