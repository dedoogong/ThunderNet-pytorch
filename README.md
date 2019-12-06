# ThunderNet-pytorch
ThunderNet-pytorch in progress. 
I will use detectron2 as a base framework. Plus, I'm considering thunder-net based segmentation like centermask or yolact.

And as far as I know, depthwise + pointwise conv is slower than reported in any related papers on NVIDIA GPUs, so I think it would be much better to apply "resnet26d" which is the fastest and accurater than a normal resnet50. 


