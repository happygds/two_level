void crop_and_resize_gpu_forward(
    THCudaTensor * features,
    THCudaTensor * rois,           // [y1, x1, y2, x2]
    const float extrapolation_value,
    const int crop_depth,
    const float temporal_scale,
    const int start_pooled_depth,
    const int end_pooled_depth,
    const float bratio,
    THCudaTensor * output
);

void crop_and_resize_gpu_backward(
    THCudaTensor * top_grad,
    THCudaTensor * rois,      // [y1, x1, y2, x2]
    const int crop_depth,
    const float temporal_scale,
    const int start_pooled_depth,
    const int end_pooled_depth,
    const float bratio,
    THCudaTensor * bottom_grad // resize to [bsize, c, hc, wc]
);
