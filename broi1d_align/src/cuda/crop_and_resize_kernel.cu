#include <math.h>
#include <stdio.h>
#include "crop_and_resize_kernel.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
     i += blockDim.x * gridDim.x)


__global__
void CropAndResizeKernel(
    const int nthreads, const float* bottom_data, const float* bottom_rois,
    int num_rois, int batch, int channels, int depth, int crop_depth,
    const float temporal_scale, const int start_pooled_depth, const int end_pooled_depth,
    const float bratio, float extrapolation_value, float* top_data)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        // (n, c, ph, pw) is an element in the pooled output
        int n = index;
        int pd = n % (crop_depth+start_pooled_depth+end_pooled_depth);
        n /= (crop_depth+start_pooled_depth+end_pooled_depth);
        int c = n % channels;
        n /= channels;
        int roi_b = n / num_rois;

        // [start, end) interval for spatial sampling
        bottom_rois += n * 3;
        int roi_batch_ind = bottom_rois[0];
        if (roi_batch_ind != roi_b)
        {
            printf("roi_batch_ind is not right !!!\n");
            return;
        }

        float roi_start = bottom_rois[1] * temporal_scale;
        float roi_end = bottom_rois[2] * temporal_scale;
        float roi_dura = roi_end - roi_start;
        // if pd in [0, start_pooled_depth)
        if (pd < start_pooled_depth)
        {
            roi_start = roi_start - bratio*roi_dura;
            roi_end = roi_start + bratio*roi_dura;
            crop_depth = start_pooled_depth;
        }
        else if (pd >= start_pooled_depth && pd < start_pooled_depth+crop_depth)
        {
            pd = pd - start_pooled_depth;
        }
        else if (pd >= start_pooled_depth+crop_depth && pd < (crop_depth+start_pooled_depth+end_pooled_depth))
        {
            roi_start = roi_end - bratio*roi_dura;
            roi_end = roi_end + bratio*roi_dura;
            pd = pd - start_pooled_depth - crop_depth;
            crop_depth = end_pooled_depth;
        }
        else
        {
            printf(" pd is not right !!!");
            return;
        }
        const float d1 = roi_start;
        const float d2 = roi_end;

        const float depth_scale =
            (crop_depth > 1) ? (d2 - d1) / (crop_depth - 1) : 0;

        const float in_d = (crop_depth > 1)
            ? d1 + pd * depth_scale
            : 0.5 * (d1 + d2);
        if (in_d <= -1 || in_d >= depth)
        {
            top_data[index] = extrapolation_value;
            continue;
        }

        const int top_d_index = floorf(in_d);
        const int bottom_d_index = ceilf(in_d);
        const float d_lerp = in_d - top_d_index;

        const float *pimage = bottom_data + (roi_batch_ind * channels + c) * depth;

        float top_out_sum = 0;
        if (in_d >= 0)
        {
            int top_index = top_d_index;
            top_out_sum += pimage[top_index];
        }

        float bottom_out_sum = 0;
        if (in_d <= depth - 1)
        {
            int bottom_index = bottom_d_index;
            bottom_out_sum += pimage[bottom_index];
        }

        top_data[index] = top_out_sum + (bottom_out_sum - top_out_sum) * d_lerp;
    }
}

__global__
void CropAndResizeBackpropImageKernel(
    const int nthreads, const float* top_grad, const float* bottom_rois,
    int num_rois, int batch, int channels, int depth, int crop_depth,
    const float temporal_scale, const int start_pooled_depth, const int end_pooled_depth,
    const float bratio, float* bottom_grad)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        // (n, c, ph, pw) is an element in the pooled output
        int n = index;
        int pd = n % crop_depth;
        n /= crop_depth;
        int c = n % channels;
        n /= channels;
        int roi_b = n / num_rois;

        // [start, end) interval for spatial sampling
        bottom_rois += n * 3;
        int roi_batch_ind = bottom_rois[0];
        if (roi_batch_ind != roi_b)
        {
            printf("roi_batch_ind is not right !!!\n");
            return;
        }
        float roi_start = bottom_rois[1] * temporal_scale;
        float roi_end = bottom_rois[2] * temporal_scale;
        float roi_dura = roi_end - roi_start;
        // if pd in [0, start_pooled_depth)
        if (pd < start_pooled_depth)
        {
            roi_start = roi_start - bratio*roi_dura;
            roi_end = roi_start + bratio*roi_dura;
            crop_depth = start_pooled_depth;
        }
        else if (pd >= start_pooled_depth && pd < start_pooled_depth+crop_depth)
        {
            pd = pd - start_pooled_depth;
        }
        else if (pd >= start_pooled_depth+crop_depth && pd < (crop_depth+start_pooled_depth+end_pooled_depth))
        {
            roi_start = roi_end - bratio*roi_dura;
            roi_end = roi_end + bratio*roi_dura;
            pd = pd - start_pooled_depth - crop_depth;
            crop_depth = end_pooled_depth;
        }
        else
        {
            printf(" pd is not right !!!");
            return;
        }
        const float d1 = roi_start;
        const float d2 = roi_end;

        const float depth_scale =
            (crop_depth > 1) ? (d2 - d1) / (crop_depth - 1) : 0;

        const float in_d = (crop_depth > 1)
            ? d1 + pd * depth_scale
            : 0.5 * (d1 + d2);
        if (in_d <= -1 || in_d >= depth)
        {
            continue;
        }

        const int top_d_index = floorf(in_d);
        const int bottom_d_index = ceilf(in_d);
        const float d_lerp = in_d - top_d_index;

        float *pimage = bottom_grad + (roi_batch_ind * channels + c) * depth;

        const float diff_val_top = top_grad[index] * (1. - d_lerp);
        const float diff_val_bottom = top_grad[index] * d_lerp;
        if (in_d >= 0)
            atomicAdd(pimage + top_d_index, diff_val_top);
        
        if (in_d <= depth - 1.)
            atomicAdd(pimage + bottom_d_index, diff_val_bottom);

    }
}


void CropAndResizeLaucher(
    const float* bottom_data, const float* bottom_rois,
    int num_rois, int batch_size, int channels, int depth,
    int crop_depth, const float temporal_scale, const int start_pooled_depth,
    const int end_pooled_depth, const float bratio, float extrapolation_value,
    float* top_data, cudaStream_t stream)
{
    const int total_count = batch_size * num_rois * (crop_depth+start_pooled_depth+end_pooled_depth) * channels;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    cudaError_t err;

    if (total_count > 0)
    {
        CropAndResizeKernel<<<block_count, thread_per_block, 0, stream>>>(
            total_count, bottom_data, bottom_rois, num_rois, batch_size, channels,
            depth, crop_depth, temporal_scale, start_pooled_depth, end_pooled_depth, 
            bratio, extrapolation_value, top_data);

        err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
            exit(-1);
        }
    }
}


void CropAndResizeBackpropImageLaucher(
    const float* top_grad, const float* bottom_rois,
    int num_rois, int batch_size, int channels, int depth,
    int crop_depth, const float temporal_scale, const int start_pooled_depth,
    const int end_pooled_depth, const float bratio,
    float* bottom_grad, cudaStream_t stream)
{
    const int total_count = batch_size * num_rois * (crop_depth+start_pooled_depth+end_pooled_depth) * channels;
    const int thread_per_block = 1024;
    const int block_count = (total_count + thread_per_block - 1) / thread_per_block;
    cudaError_t err;

    if (total_count > 0)
    {
        CropAndResizeBackpropImageKernel<<<block_count, thread_per_block, 0, stream>>>(
            total_count, top_grad, bottom_rois, num_rois, batch_size, channels,
            depth, crop_depth, temporal_scale, start_pooled_depth, end_pooled_depth, 
            bratio, bottom_grad);

        err = cudaGetLastError();
        if (cudaSuccess != err)
        {
            fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
            exit(-1);
        }
    }
}
