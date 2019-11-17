from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from thundernet.utils.common import DepthwiseConv5x5, Conv1x1
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from roi_align.crop_and_resize import CropAndResizeFunction

class PSRoiAlignPooling(Function):
    def __init__(self, pooled_height, pooled_width, spatial_scale, group_size, output_dim):
        self.pooled_width = int(pooled_width)
        self.pooled_height = int(pooled_height)
        self.spatial_scale = float(spatial_scale)
        self.group_size = int(group_size)
        self.output_dim = int(output_dim)
        self.output = None
        self.mappingchannel = None
        self.rois = None
        self.feature_size = None

    def forward(self, features, rois):
        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size()[0]
        output = torch.zeros(num_rois, self.output_dim, self.pooled_height, self.pooled_width)
        mappingchannel = torch.IntTensor(num_rois, self.output_dim, self.pooled_height, self.pooled_width).zero_()
        output = output.cuda()
        mappingchannel = mappingchannel.cuda()
        '''
        psroi_pooling.psroi_pooling_forward_cuda(self.pooled_height, self.pooled_width, self.spatial_scale,
                                                 self.group_size, self.output_dim, \
                                             features, rois, output, mappingchannel);
        '''
        self.output = output
        self.mappingchannel = mappingchannel
        self.rois = rois
        self.feature_size = features.size()

        return output
    '''
    def backward(self, grad_output):
        assert (self.feature_size is not None and grad_output.is_cuda)

        batch_size, num_channels, data_height, data_width = self.feature_size

        grad_input = torch.zeros(batch_size, num_channels, data_height, data_width).cuda()

        psroi_pooling.psroi_pooling_backward_cuda(self.pooled_height, self.pooled_width, self.spatial_scale,
                                                  self.output_dim, \
                                                  grad_output, self.rois, grad_input, self.mappingchannel)
        return grad_input, None
    '''

class PSRoiAlignPooling(nn.Module):
    """ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img,X_roi] with shape:
        X_img:
        `(1, rows, cols, channels)`
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    """

    def __init__(self, pool_size, num_rois, alpha, **kwargs):
        self.dim_ordering = 'tf'
        self.pool_size = pool_size
        self.num_rois = num_rois
        self.alpha_channels = alpha

        super(PSRoiAlignPooling).__init__()

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.alpha_channels

    def forward(self, x, mask=None):
        assert (len(x) == 2)
        total_bins = 1
        # x[0] is image with shape (rows, cols, channels)
        img = x[0]

        # x[1] is roi with shape (num_rois,4) with ordering (x,y,w,h)
        rois = x[1]

        # because crop_size of tf.crop_and_resize requires 1-D tensor, we use uniform length
        bin_crop_size = []
        for num_bins, crop_dim in zip((7, 7), (14, 14)):
            assert num_bins >= 1
            assert crop_dim % num_bins == 0
            total_bins *= num_bins
            bin_crop_size.append(crop_dim // num_bins)

        xmin, ymin, xmax, ymax = torch.unbind(rois[0], dim=1) # torch.tensor([[1, 2, 3],[4, 5, 6],[7, 8, 9]]))
                                                              # ->(tensor([1, 2, 3]), tensor([4, 5, 6]), tensor([7, 8, 9]))
        spatial_bins_y =  spatial_bins_x = 7
        step_y = (ymax - ymin) / spatial_bins_y
        step_x = (xmax - xmin) / spatial_bins_x

        # gen bins
        position_sensitive_boxes = []
        for bin_x in range(self.pool_size): 
            for bin_y in range(self.pool_size):
                box_coordinates = [
                    ymin + bin_y * step_y,
                    xmin + bin_x * step_x,
                    ymin + (bin_y + 1) * step_y,
                    xmin + (bin_x + 1) * step_x 
                ]
                position_sensitive_boxes.append(torch.stack(box_coordinates, dim=1))
        
        img_splits = torch.split(img, total_bins, dim=3)
        box_image_indices = np.zeros(self.num_rois)

        feature_crops = []
        for split, box in zip(img_splits, position_sensitive_boxes):
            #assert box.shape[0] == box_image_indices.shape[0], "Psroi box number doesn't match roi box indices!"
            #crop = tf.image.crop_and_resize(
            #    split, box, box_image_indices,
            #    bin_crop_size, method='bilinear'
            #)
            crop = CropAndResizeFunction.apply(split, box, box_image_indices, bin_crop_size[0], bin_crop_size[1], 0)
            # shape [num_boxes, crop_height/spatial_bins_y, crop_width/spatial_bins_x, depth/total_bins]

            # do max pooling over spatial positions within the bin
            crop_1 = torch.max(crop, dim=1, keepdim=False, out=None) #tf.reduce_max(crop, axis=[1, 2])
            crop_2 = torch.max(crop, dim=2, keepdim=False, out=None)  # tf.reduce_max(crop, axis=[1, 2])
            crop = torch.stack(crop_1, crop_2)
            crop = crop.unsqueeze(1) #tf.expand_dims(crop, 1)
            # shape [num_boxes, 1, depth/total_bins]

            feature_crops.append(crop)

        final_output = torch.cat(feature_crops, dim=1)

        # Reshape to (1, num_rois, pool_size, pool_size, nb_channels)
        # Might be (1, 4, 7, 7, 5)
        final_output = final_output.reshape(1, self.num_rois, self.pool_size, self.pool_size, self.alpha_channels)

        # permute_dimensions is similar to transpose
        final_output = final_output.permute(0, 1, 2, 3, 4)

        return final_output

class RPN(nn.Module):

    def __init__(self,in_channels, num_anchors, nb_classes, in_channels2):#==base_layers.shape[3], num_anchors, nb_classes, x.shape[3]
        super(RPN).__init__()
        #rpn part
        self.conv1x1 = Conv1x1(in_channels=in_channels, out_channels=245, strides=1, groups=1)  # use_bias=True, name='sam/conv1x1')
        self.depthwise_conv5x5 = DepthwiseConv5x5(channels=245,strides=1) #'rpn/conv5x5'
        self.conv1x1 = Conv1x1(in_channels=in_channels2, out_channels=256, strides=1, groups=1)# use_bias=True, 'rpn/conv1x1'
        self.conv2   = nn.Conv2d(num_anchors, (1, 1))
        self.sigmoid = nn.Sigmoid() # kernel_initializer='uniform')#'rpn_out_class'
        self.conv3   = nn.Conv2d(num_anchors * 4, (1, 1)) # , activation='linear', kernel_initializer='zero') #'rpn_out_regress'

        # classifier part
        self.dropout = nn.Dropout(0.5)
        self.batchnorm = nn.BatchNorm2d()
        self.linear = nn.Linear(1024)
        self.linear_cls = nn.Linear(nb_classes)#, activation='softmax', kernel_initializer='zero'),
        self.softmax = nn.Softmax()
        self.linear_reg = nn.Linear(4 * (nb_classes - 1)) # activation='linear', kernel_initializer='zero'),

    def forward(self, x):
        """Create a rpn layer
            Step1: Pass through the feature map from base layer to a 256 channels convolutional layer
                    Keep the padding 'same' to preserve the feature map's size
            Step2: Pass the step1 to two (1,1) convolutional layer to replace the fully connected layer
                    classification layer: num_anchors (9 in here) channels for 0, 1 sigmoid activation output
                    regression layer: num_anchors*4 (36 here) channels for regression of bboxes with linear activation
        Args:
            base_layers: snet in here
            num_anchors: 9 in here

        Returns:
            [x_class, x_regr, base_layers]
            x_class: classification for whether it's an object
            x_regr: bboxes regression
            base_layers: snet in here
        """
        x=self.depthwise_conv5x5(x)
        x=self.conv1x1(x)
        x_class=self.sigmoid(self.conv2(x))
        x_regr =self.conv3(x)
        return [x_class, x_regr] # , base_layers]

    def classifier(self, base_layers, input_rois, num_rois, nb_classes=3):
        """Create a classifier layer

        Args:
            base_layers: snet
            input_rois: `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
            num_rois: number of rois to be processed in one time (4 in here)
            nb_classes: default number of classes

        Returns:
            list(out_class, out_regr)
            out_class: classifier layer output
            out_regr: regression layer output
        """

        x = self.conv1x1(base_layers)
        x = self.batchnorm(x) #name='sam/bn'
        x = self.sigmoid(x)
        x = x*base_layers

        pooling_regions = 7
        alpha = 5

        # out_roi_pool.shape = (1, num_rois, channels, pool_size, pool_size)
        # num_rois (4) 7x7 roi pooling
        # out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([x, input_rois])
        out_roi_pool = PSRoiAlignPooling(pooling_regions, num_rois, alpha)([x, input_rois])

        # Flatten the convlutional layer and connected to 2 FC and 2 dropout
        out = torch.flatten(out_roi_pool)#TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
        out = self.linear(out)           #TimeDistributed(Dense(1024, activation='relu', name='fc'))(out)
        out = self.dropout(out)          #TimeDistributed(Dropout(0.5))(out)

        # There are two output layer
        out_score = self.linear_cls(out)
        out_class = self.softmax(out_score)

        # note: no regression target for bg class
        out_regr = self.linear_reg(out)

        return [out_class, out_regr]
