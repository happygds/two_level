#include <THC/THC.h>
#include "cuda/crop_and_resize_kernel.h"

extern THCState *state;


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
) {
    // Grab the input tensor
    float * data_flat = THCudaTensor_data(state, features);
    float * rois_flat = THCudaTensor_data(state, rois);

    float * output_flat = THCudaTensor_data(state, output);

    // Number of ROIs
    const int num_rois = THCudaTensor_size(state, rois, 1);
    const int size_rois = THCudaTensor_size(state, rois, 2);
    if (size_rois != 3)
    {
        return;
    }

    // batch size
    const int batch_size = THCudaTensor_size(state, features, 0);
    // Number of channels
    const int num_channels = THCudaTensor_size(state, features, 1);
    // data depth
    const int data_depth = THCudaTensor_size(state, features, 2);

    cudaStream_t stream = THCState_getCurrentStream(state);
    CropAndResizeLaucher(
        data_flat, rois_flat, num_rois, batch_size,
        num_channels, data_depth, crop_depth, temporal_scale,
        start_pooled_depth, end_pooled_depth, bratio,
        extrapolation_value, output_flat, stream
    );
    return;
}


void crop_and_resize_gpu_backward(
    THCudaTensor * top_grad,
    THCudaTensor * rois,      // [y1, x1, y2, x2]
    const int crop_depth,
    const float temporal_scale,
    const int start_pooled_depth,
    const int end_pooled_depth,
    const float bratio,
    THCudaTensor * bottom_grad // resize to [bsize, c, hc, wc]
) {
    // Grab the input tensor
    float * top_grad_flat = THCudaTensor_data(state, top_grad);
    float * rois_flat = THCudaTensor_data(state, rois);

    float * bottom_grad_flat = THCudaTensor_data(state, bottom_grad);

    // Number of ROIs
    const int num_rois = THCudaTensor_size(state, rois, 1);
    const int size_rois = THCudaTensor_size(state, rois, 2);
    if (size_rois != 3)
    {
        return;
    }

    // batch size
    const int batch_size = THCudaTensor_size(state, bottom_grad, 0);
    // Number of channels
    const int num_channels = THCudaTensor_size(state, bottom_grad, 1);
    // data depth
    const int data_depth = THCudaTensor_size(state, bottom_grad, 2);

    cudaStream_t stream = THCState_getCurrentStream(state);
    CropAndResizeBackpropImageLaucher(
        top_grad_flat, rois_flat, num_rois, batch_size,
        num_channels, data_depth, crop_depth, temporal_scale,
        start_pooled_depth, end_pooled_depth, bratio,
        bottom_grad_flat, stream
    );
    return;
}
