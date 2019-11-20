# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.modules.utils import _pair

from thundernet import _C

''' 
class ROIPooler(nn.Module): 

    def __init__(
        self,
        output_size,
        scales,
        sampling_ratio,
        pooler_type,
        canonical_box_size=224,
        canonical_level=4,
    ):
         
    Args:
        output_size (int, tuple[int] or list[int]): output size of the pooled region,
            e.g., 14 x 14. If tuple or list is given, the length must be 2.
        scales (list[float]): The scale for each low-level pooling op relative to
            the input image. For a feature map with stride s relative to the input
            image, scale is defined as a 1 / s. The stride must be power of 2.
            When there are multiple scales, they must form a pyramid, i.e. they must be
            a monotically decreasing geometric sequence with a factor of 1/2.
        sampling_ratio (int): The `sampling_ratio` parameter for the ROIAlign op.
        pooler_type (string): Name of the type of pooling operation that should be applied.
            For instance, "ROIPool" or "ROIAlignV2".
        canonical_box_size (int): A canonical box size in pixels (sqrt(box area)). The default
            is heuristically defined as 224 pixels in the FPN paper (based on ImageNet
            pre-training).
        canonical_level (int): The feature map level index from which a canonically-sized box
            should be placed. The default is defined as level 4 (stride=16) in the FPN paper,
            i.e., a box of size 224x224 will be placed on the feature with stride=16.
            The box placement for all boxes will be determined from their sizes w.r.t
            canonical_box_size. For example, a box whose area is 4x that of a canonical box
            should be used to pool features from feature level ``canonical_level+1``.

            Note that the actual input feature maps given to this module may not have
            sufficiently many levels for the input boxes. If the boxes are too large or too
            small for the input feature maps, the closest level will be used.  

    self.level_poolers = nn.ModuleList( ROIAlign(
                        output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=True)
                        for scale in scales )
'''
class _ROIAlign(Function):
    @staticmethod
    def forward(ctx, input, roi, output_size, spatial_scale, sampling_ratio, aligned):
        ctx.save_for_backward(roi)
        ctx.output_size = _pair(output_size)
        ctx.spatial_scale = spatial_scale
        ctx.sampling_ratio = sampling_ratio
        ctx.input_shape = input.size()
        ctx.aligned = aligned
        output = _C.roi_align_forward( input, roi, spatial_scale, output_size[0], output_size[1], sampling_ratio, aligned )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        output_size = ctx.output_size
        spatial_scale = ctx.spatial_scale
        sampling_ratio = ctx.sampling_ratio
        bs, ch, h, w = ctx.input_shape
        grad_input = _C.roi_align_backward( grad_output,
                                            rois,
                                            spatial_scale,
                                            output_size[0],
                                            output_size[1],
                                            bs,
                                            ch,
                                            h,
                                            w,
                                            sampling_ratio,
                                            ctx.aligned, )
        return grad_input, None, None, None, None, None


roi_align = _ROIAlign.apply


class ROIAlign(nn.Module):
    def __init__(self, output_size, spatial_scale, sampling_ratio, aligned=True):
        """
        Args:
            output_size (tuple): h, w
            spatial_scale (float): scale the input boxes by this number
            sampling_ratio (int): number of inputs samples to take for each output
                                  sample. 0 to take samples densely.
            aligned (bool): if False, use the legacy implementation in Detectron.
                            If True, align the results more perfectly.

        Note:
            The meaning of aligned=True:

            Given a continuous coordinate c, its two neighboring pixel indices (in our
            pixel model) are computed by floor(c - 0.5) and ceil(c - 0.5). For example,
            c=1.3 has pixel neighbors with discrete indices [0] and [1] (which are sampled
            from the underlying signal at continuous coordinates 0.5 and 1.5). But the original
            roi_align (aligned=False) does not subtract the 0.5 when computing neighboring
            pixel indices and therefore it uses pixels with a slightly incorrect alignment
            (relative to our pixel model) when performing bilinear interpolation.

            With `aligned=True`,
            we first appropriately scale the ROI and then shift it by -0.5
            prior to calling roi_align. This produces the correct neighbors; see
            detectron2/tests/test_roi_align.py for verification.

            The difference does not make a difference to the model's performance if
            ROIAlign is used together with conv layers.
        """
        super(ROIAlign, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        self.sampling_ratio = sampling_ratio
        self.aligned = aligned

    def forward(self, input, rois):
        """
        Args:
            input: NCHW images
            rois: Bx5 boxes. First column is the index into N. The other 4 columns are xyxy.
        """
        assert rois.dim() == 2 and rois.size(1) == 5
        return roi_align(input, rois, self.output_size, self.spatial_scale, self.sampling_ratio, self.aligned)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "output_size=" + str(self.output_size)
        tmpstr += ", spatial_scale=" + str(self.spatial_scale)
        tmpstr += ", sampling_ratio=" + str(self.sampling_ratio)
        tmpstr += ", aligned=" + str(self.aligned)
        tmpstr += ")"
        return tmpstr
