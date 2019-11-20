from thundernet.utils.common import Conv1x1, DepthwiseConv5x5, Conv1x1Block, Conv3x3Block #, MaxPool2D, \
                                    #Channel_shuffle_lambda, se_block, batchnorm, is_channels_first, get_channel_axis
import torch
import torch.nn as nn
from torch.nn import functional as F

CEM_FILTER=245
# INPUT : N H W C == 10 320 320 3
class CEM(nn.Module):#Model):
    """Context Enhancement Module"""
    def __init__(self):
        super(CEM, self).__init__()
        self.conv4 = nn.Conv2d(out_channels=CEM_FILTER, kernel_size=1, stride=1, padding="SAME") #, use_bias=True)
        self.conv5 = nn.Conv2d(out_channels=CEM_FILTER, kernel_size=1, stride=1, padding="SAME") #, use_bias=True)
        self.conv6 = nn.Conv2d(out_channels=CEM_FILTER, kernel_size=1, stride=1, padding="SAME") #, use_bias=True)
        self.upsample = nn.Upsample(scale_factor=(2,2), mode="bilinear")
        #self.b = K.reshape(inputs, [-1, h, w, in_channel // group, group])

    #@tf.function TF uses NHWC, torch uses NCHW
    def forward(self, inputs, training=False):
        C4_lat = self.conv4(inputs[0])
        C5_lat = self.conv5(inputs[1])
        C5_lat = F.interpolate(input=C5_lat,scale_factor=(2,2),mode="bilinear" )
        # tf.keras.backend.resize_images(x=C5_lat, height_factor=2, width_factor=2, data_format="channels_last", interpolation="bilinear")

        # wrong?? torch.reshape(inputs[2], [-1, 1, 1, CEM_FILTER]) #K.reshape(inputs[2], [-1, 1, 1, CEM_FILTER])
        Cglb_lat = self.conv6(inputs[2]) # x == FC(GAP(C5)), FC== Dense(num_class) !!

        return C4_lat + C5_lat + Cglb_lat
'''    
#wrong for x2 !! conv1x1 first and then upsamplex2 !!
def context_enhancement_module(x1, x2, x3, size=20): #, name='cem_block'):
    x1 = Conv1x1(x1,in_channels=x1.shape[3], out_channels=CEM_FILTER, strides=1, groups=1)

    x2 = Conv1x1(x2, in_channels=x2.shape[3], out_channels=CEM_FILTER, strides=1, groups=1)
    x2 = F.interpolate(x2, size=[size, size], align_corners=True)
    # tf.image.resize_bilinear(img, [20, 20],align_corners=True,
    #                       name='{}/c5_resize'.format(name))(x2)

    x3 = Conv1x1(x3, in_channels=x3.shape[3], out_channels=CEM_FILTER, strides=1, groups=1) # N, 245, 1, 1
    x3 = x3 + torch.zeros((1, size, size, 528)) #nn.Lambda(lambda img: nn.add([img, zero]))(x3)

    return x1 + x2 + x3 # -> 20X20X245
'''
class SAM(nn.Module): #(Model):
    """spatial attention module"""
    def __init__(self):
        super(SAM, self).__init__()
        self.point = nn.Conv2d(out_channels=CEM_FILTER, kernel_size=1, stride=1, padding="VALID") #, use_bias=False)
        self.bn = nn.BatchNorm2d(momentum=0.9, eps=1e-5) #BatchNormalization()
        self.softmax = nn.Softmax(1)

    def forward(self, inputs): #:, training=False):
        #inputs==[RPN, CEM]
        x = self.point(inputs[0])#==output of RPN
        x = self.bn(x)
        x = self.softmax(x) #tf.keras.activations.softmax(x, axis=-1) why not sigmoid??
        x = x*inputs[1] #tf.math.multiply(x, inputs[1]==output of CEM)
        return x

class ShuffleInitBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ShuffleInitBlock).__init__()
        self.conv3x3_block = Conv3x3Block(in_channels=in_channels, out_channels=out_channels, strides=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

    def forward(self, x, in_channels, out_channels):
        x = self.conv3x3_block(x)
        x = self.maxpool(x)
        return x

class ShuffleUnit(nn.Module):

    def __init__(self, in_channels, out_channels, downsample,
                 use_se, use_residual):  # name="shuffle_unit"):
        super(ShuffleUnit).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample = downsample
        self.use_se = use_se
        self.use_residual = use_residual
        self.batchnorm = nn.BatchNorm2d(momentum=0.9, eps=1e-5)

        self.mid_channels = self.out_channels // 2
        self.depthconv_5x5 = DepthwiseConv5x5(channels=self.in_channels, stride=2)  # name=name + "/dw_conv4")
        self.conv1x1_downsample = Conv1x1(in_channels=in_channels,
                                          out_channels=self.mid_channels)  # name=name + "/expand_conv5")
        self.in_split2_channels = self.in_channels // 2

        self.no_downsample_y1 = lambda x: x[:, 0:self.in_split2_channels, :, :]
        self.no_downsample_x2 = lambda x: x[:, self.in_split2_channels:, :, :]

        self.depthwise_conv5x5_2 = DepthwiseConv5x5(channels=self.mid_channels)

        strides = (2 if downsample else 1),
        self.conv1x1_1 = Conv1x1(in_channels=(self.in_channels if self.downsample else self.mid_channels),
                               out_channels=self.mid_channels)
        self.conv1x1_2 = Conv1x1(in_channels=self.mid_channels, out_channels=self.mid_channels) # name=name + "/expand_conv3")

    def forward(self, x):
        if self.downsample:
            y1 = self.depthconv_5x5(x)
            y1 = self.batchnorm(y1)
            y1 = self.conv1x1_downsample(y1)
            y1 = self.batchnorm(y1)  # name=name + "/expand_bn5")
            y1 = F.relu(y1)
            x2 = x
        else:
            y1 = self.no_downsample_y1(x)  # nn.Lambda(lambda z: z[:, 0:in_split2_channels, :, :])(x)
            x2 = self.no_downsample_x2(x)  # nn.Lambda(lambda z: z[:, in_split2_channels:, :, :])(x)
            '''
            if is_channels_first():
                y1 = lambda x: x[:,0:in_split2_channels, :, :] #nn.Lambda(lambda z: z[:, 0:in_split2_channels, :, :])(x)
                x2 = lambda x: x[:,  in_split2_channels:,:, :] #nn.Lambda(lambda z: z[:, in_split2_channels:, :, :])(x)
            else:
                y1 = lambda z: z[:, :, :, 0:in_split2_channels] #nn.Lambda(lambda z: z[:, :, :, 0:in_split2_channels])(x)
                x2 = lambda z: z[:, :, :,   in_split2_channels:]#nn.Lambda(lambda z: z[:, :, :, in_split2_channels:])(x)
            '''
        # name=name + "/compress_conv1")
        y2 = self.Conv1x1_1(x2)
        y2 = self.batchnorm(y2)  # name=name + "/compress_bn1")
        y2 = F.ReLU(y2)
        y2 = self.depthwise_conv5x5_2(y2)  # name=name + "/dw_conv2")
        y2 = self.batchnorm(y2)  # name=name + "/dw_bn2")

        y2 = self.conv1x1_2(y2)

        y2 = self.batchnorm(y2) # name=name + "/expand_bn3")
        y2 = F.ReLU(y2)
        '''
        if self.use_se:
            y2 = se_block(
                x=y2,
                channels=mid_channels,
                name=name + "/se")
    
        if self.use_residual and not self.downsample:
            y2 = y2 + x2  # nn.add([y2, x2], name=name + "/add")
    
        x = torch.cat([y1, y2], dim=1) #get_channel_axis())  # nn.concatenate([y1, y2], axis=get_channel_axis(), name=name + "/concat")
    
        x = channel_shuffle_lambda(
            channels=out_channels,
            groups=2,
            name=name + "/c_shuffle")(x)
        '''
        return x


class ShuffleNetV2(nn.Module):

    def __init__(self,channels,init_block_channels,final_block_channels,in_channels=3,
                 use_se=False, use_residual=False,in_size=(320, 320),classes=2):#model_name="snet_146"):
        super(ShuffleNetV2).__init__()
        # input_shape = (in_channels, 320, 320) if is_channels_first() else (320, 320, in_channels)
        # input = nn.Input(shape=input_shape)

        self.shuffle_init_block =ShuffleInitBlock(in_channels=in_channels,out_channels=init_block_channels)# name="features/init_block")
        self.in_channels = init_block_channels
        self.channels = channels
        self.use_se=use_se
        self.use_residual=use_residual
        self.in_size=in_size
        self.classes=classes
        '''
        self.shuffle_unit = ShuffleUnit(in_channels= self.in_channels,
                                        out_channels=out_channels,
                                        downsample=downsample,
                                        use_se=self.use_se,
                                        use_residual=self.use_residual)#name="features/stage{}/unit{}".format(i + 2, j + 1)))
        '''


    def forward(self,x):
        x = self.shuffle_init_block(x)

        count_stage = 1
        for i, channels_per_stage in enumerate(self.channels):
            for j, out_channels in enumerate(channels_per_stage):
                downsample = (j == 0)
                x = ShuffleUnit(x)

                print(x.shape)
                in_channels = out_channels
            count_stage += 1
            if count_stage == 3:
                c4 = x
            elif count_stage == 4:
                c5 = x

        if 0: #model_name == 'snet_49':
            x = conv1x1_block(
                x=x,
                in_channels=self.in_channels,
                out_channels=final_block_channels,
                name="features/final_block")
            in_channels = final_block_channels
            # print(in_channels)
        else:
            in_channels = 1
        #       nn.avg_pool2d(x, x.size()[2:]) works fine when x.shape=N * C * H * W


        c_glb = F.avg_pool2d(x, x.size()[2:])#x, x.size()[2:])  # name="features/final_pool" #self.gap(x)

        # x = flatten(c_glb)
        # x = nn.Dense(
        #     units=classes,
        #     input_dim=in_channels,
        #     name="output")(x)

        y_cem = context_enhancement_module(x1=c4,
                                           x2=c5,
                                           x3=c_glb,
                                           size=20)
        return y_cem

