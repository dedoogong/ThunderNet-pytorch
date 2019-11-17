import math


class Config:

    def __init__(self):
        # Print the process or not
        self.verbose = True

        # Feature extract network, in the paper use snet-146
        self.network = 'snet'

        # Setting for data augmentation
        self.use_horizontal_flips = False
        self.use_vertical_flips = False
        self.rot_90 = False

        # Anchor box scales
        # Original anchor_box_scales in the paper is [32, 64, 128, 256, 512]
        self.anchor_box_scales = [64, 128, 256]

        # Anchor box ratios
        # Original anchor_box_ratios in the paper is [1:2, 3:4, 1:1, 4:3, 2:1]
        self.anchor_box_ratios = [[1, 1], [1./math.sqrt(2), 2./math.sqrt(2)], [2./math.sqrt(2), 1./math.sqrt(2)]]

        # Paper request input size is 320x320
        self.im_size = 320

        # image channel-wise mean to subtract
        self.img_channel_mean = [103.939, 116.779, 123.68]
        self.img_scaling_factor = 1.0

        # number of ROIs at once
        self.num_rois = 4

        # stride at the RPN
        self.rpn_stride = 16

        self.balanced_classes = False

        # scaling the stdev
        self.std_scaling = 4.0
        self.classifier_regr_std = [8.0, 8.0, 4.0, 4.0]

        # overlaps for RPN
        self.rpn_min_overlap = 0.3
        self.rpn_max_overlap = 0.7

        # overlaps for classifier ROIs
        self.classifier_min_overlap = 0.1
        self.classifier_max_overlap = 0.5

        # placeholder for the class mapping, automatically generated by the parser
        self.class_mapping = None

        self.model_path = None