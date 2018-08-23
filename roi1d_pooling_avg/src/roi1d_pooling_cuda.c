#include <THC/THC.h>
#include <math.h>
#include "cuda/roi1d_pooling_kernel.h"

extern THCState *state;

int roi1d_pooling_forward_cuda(int pooled_depth, float temporal_scale,
                        THCudaTensor * features, THCudaTensor * rois, THCudaTensor * output)
{
    // Grab the input tensor
    float * data_flat = THCudaTensor_data(state, features);
    float * rois_flat = THCudaTensor_data(state, rois);

    float * output_flat = THCudaTensor_data(state, output);

    // Number of ROIs
    int num_rois = THCudaTensor_size(state, rois, 1);
    int size_rois = THCudaTensor_size(state, rois, 2);
    if (size_rois != 3)
    {
        return 0;
    }

    // batch size
    int batch_size = THCudaTensor_size(state, features, 0);
    // Number of channels
    int num_channels = THCudaTensor_size(state, features, 1);
    // data depth
    int data_depth = THCudaTensor_size(state, features, 2);

    cudaStream_t stream = THCState_getCurrentStream(state);

    ROI1DPoolForwardLaucher(
        data_flat, temporal_scale, num_rois,
        data_depth, num_channels,
        pooled_depth, rois_flat,
        batch_size, output_flat, stream);

    return 1;
}

int roi1d_pooling_backward_cuda(int pooled_depth, float temporal_scale,
                        THCudaTensor * top_grad, THCudaTensor * rois, THCudaTensor * bottom_grad)
{
    // Grab the input tensor
    float * top_grad_flat = THCudaTensor_data(state, top_grad);
    float * rois_flat = THCudaTensor_data(state, rois);

    float * bottom_grad_flat = THCudaTensor_data(state, bottom_grad);

    // Number of ROIs
    int num_rois = THCudaTensor_size(state, rois, 1);
    int size_rois = THCudaTensor_size(state, rois, 2);
    if (size_rois != 3)
    {
        return 0;
    }

    // batch size
    int batch_size = THCudaTensor_size(state, bottom_grad, 0);
    // Number of channels
    int num_channels = THCudaTensor_size(state, bottom_grad, 1);
    // data depth
    int data_depth = THCudaTensor_size(state, bottom_grad, 2);

    cudaStream_t stream = THCState_getCurrentStream(state);
    ROI1DPoolBackwardLaucher(
        top_grad_flat, temporal_scale, batch_size,
        num_rois, data_depth,
        num_channels, pooled_depth,
        rois_flat, bottom_grad_flat, stream);

    return 1;
}
