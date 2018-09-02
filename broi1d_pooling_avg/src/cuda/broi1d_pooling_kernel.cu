#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <math.h>
#include <float.h>
#include "broi1d_pooling_kernel.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

// NCDHW format
__global__ void BROI1DPoolForward(
    const int nthreads, const float* bottom_data, const float temporal_scale, const int num_rois, 
    const int depth, const int channels, const int pooled_depth, const int start_pooled_depth, 
    const int end_pooled_depth, const float bratio, const float* bottom_rois, float* top_data)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        // (n, c, ph, pw) is an element in the pooled output
        int n = index;
        int pd = n % (pooled_depth+start_pooled_depth+end_pooled_depth);
        n /= (pooled_depth+start_pooled_depth+end_pooled_depth);
        int c = n % channels;
        n /= channels;
        int roi_b = n / num_rois;

        // [start, end) interval for spatial sampling
        bottom_rois += n * 3;
        int roi_batch_ind = bottom_rois[0];
        // int roi_batch_ind = n / num_rois;
        float roi_start = bottom_rois[1] * temporal_scale;
        float roi_end = bottom_rois[2] * temporal_scale;
        float roi_dura = roi_end - roi_start;
        // if pd in [0, start_pooled_depth)
        if (pd < start_pooled_depth)
        {
            int roi_start_d = round(roi_start - bratio*roi_dura);
            int roi_end_d = round(roi_start + bratio*roi_dura);
        }
        elif (pd >= start_pooled_depth && pd < start_pooled_depth+pooled_depth)
        {
            int roi_start_d = round(roi_start);
            int roi_end_d = round(roi_end);
            pd = pd - start_pooled_depth;
        }
        elif (pd >= start_pooled_depth+pooled_depth && pd < (pooled_depth+start_pooled_depth+end_pooled_depth))
        {
            int roi_start_d = round(roi_end - bratio*roi_dura);
            int roi_end_d = round(roi_end + bratio*roi_dura);
            pd = pd - start_pooled_depth - pooled_depth;
        }
        else
        {
            printf(" pd is not right !!!");
            exit(-1);
            return;
        }
        if (roi_end_d > roi_start_d)
        {   
            if (roi_b != roi_batch_ind)
            {
                printf("n=%d, channels=%d, c=%d, num_rois=%d, index=%d, roi_batch_ind=%d, roi_b=%d,\
                roi_start_d=%d, roi_end_d=%d, pooled_depth=%d, *bottom_rois=%f\n", 
                n, channels, c, num_rois, index, roi_batch_ind, roi_b, roi_start_d, roi_end_d, pooled_depth, *bottom_rois);
                exit(-1);
                return;
            }
    
            // Force malformed ROIs to be 1x1
            int roi_depth = max(roi_end_d - roi_start_d, 1);
            float bin_size_d = (float)(roi_depth) / (float)(pooled_depth);
    
            int dstart = (int)(floor((float)(pd) * bin_size_d));
            int dend = (int)(ceil((float)(pd + 1) * bin_size_d));
            float bin_area = dend - dstart;
    
            // Add roi offsets and clip to input boundaries
            dstart = fminf(fmaxf(dstart + roi_start_d, 0), depth);
            dend = fminf(fmaxf(dend + roi_start_d, 0), depth);
            bool is_empty = dend <= dstart;
    
            // Define an empty pooling region to be zero
            float out_sum = 0;
            // float maxval = is_empty ? 0 : -FLT_MAX;
            // // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
            // int maxidx = -1;
            bottom_data += (roi_batch_ind * channels + c) * depth;
            for (int d = dstart; d < dend; ++d) {
                out_sum += bottom_data[d];
                // if (bottom_data[bottom_index] > maxval) {
                //     maxval = bottom_data[bottom_index];
                //     maxidx = bottom_index;
                // }
            }
            // top_data[index] = maxval;
            // if (argmax_data != NULL)
            //     argmax_data[index] = maxidx;
            top_data[index] = is_empty? 0. : out_sum/bin_area;
        }
        else
        {
            top_data[index] = 0.;
        }
    }
}

int BROI1DPoolForwardLaucher(
    const float* bottom_data, const float temporal_scale, const int num_rois, const int depth, const int channels, 
    const int pooled_depth, const int start_pooled_depth, const int end_pooled_depth, const float bratio, 
    const float* bottom_rois, const int batch_size, float* top_data, cudaStream_t stream)
{
    const int kThreadsPerBlock = 1024;
    const int output_size = batch_size * num_rois * (pooled_depth+start_pooled_depth+end_pooled_depth) * channels;
    cudaError_t err;

    BROI1DPoolForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
      output_size, bottom_data, temporal_scale, num_rois, depth, channels,
      pooled_depth, start_pooled_depth, end_pooled_depth, bratio, bottom_rois, top_data);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}


__global__ void BROI1DPoolBackward(
    const int nthreads, const float* top_diff, const int num_rois, const float temporal_scale,
    const int depth,  const int channels, const int pooled_depth, const int start_pooled_depth, 
    const int end_pooled_depth, const float bratio, float* bottom_diff, const float* bottom_rois) {
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        // (n, c, ph, pw) is an element in the pooled output
        int n = index;
        int pd = n % (pooled_depth+start_pooled_depth+end_pooled_depth);
        n /= (pooled_depth+start_pooled_depth+end_pooled_depth);
        int c = n % channels;
        n /= channels;
        int roi_b = n / num_rois;

        // [start, end) interval for spatial sampling
        bottom_rois += n * 3;
        int roi_batch_ind = bottom_rois[0];
        // int roi_batch_ind = n / num_rois;
        float roi_start = bottom_rois[1] * temporal_scale;
        float roi_end = bottom_rois[2] * temporal_scale;
        float roi_dura = roi_end - roi_start;
        // if pd in [0, start_pooled_depth)
        if (pd < start_pooled_depth)
        {
            int roi_start_d = round(roi_start - bratio*roi_dura);
            int roi_end_d = round(roi_start + bratio*roi_dura);
        }
        elif (pd >= start_pooled_depth && pd < start_pooled_depth+pooled_depth)
        {
            int roi_start_d = round(roi_start);
            int roi_end_d = round(roi_end);
            pd = pd - start_pooled_depth;
        }
        elif (pd >= start_pooled_depth+pooled_depth && pd < (pooled_depth+start_pooled_depth+end_pooled_depth))
        {
            int roi_start_d = round(roi_end - bratio*roi_dura);
            int roi_end_d = round(roi_end + bratio*roi_dura);
            pd = pd - start_pooled_depth - pooled_depth;
        }
        else
        {
            printf(" pd is not right !!!");
            exit(-1);
        }
        if (roi_end_d > roi_start_d)
        {
            if (roi_batch_ind != roi_b)
            {
                printf("roi_batch_ind is not right !!!\n");
                exit( -1 );
            }
    
            // Force malformed ROIs to be 1x1
            int roi_depth = max(roi_end_d - roi_start_d + 1, 1);
            float bin_size_d = (float)(roi_depth) / (float)(pooled_depth);
    
            int dstart = (int)(floor((float)(pd) * bin_size_d));
            int dend = (int)(ceil((float)(pd + 1) * bin_size_d));
            float bin_area = max(dend - dstart, 1);
    
            // Add roi offsets and clip to input boundaries
            dstart = fminf(fmaxf(dstart + roi_start_d, 0), depth);
            dend = fminf(fmaxf(dend + roi_start_d, 0), depth);
            bool is_empty = dend <= dstart || dend <= 0;
    
            // Compute c at bottom
            float* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * depth;
            // float bin_area = (hend - hstart) * (wend - wstart) * (dend - dstart);
            float diff_val = is_empty ? 0. : top_diff[index] / bin_area;
            for (int d = dstart; d < dend; ++d) {
                //caffe_gpu_atomic_add(diff_val, offset_bottom_diff + bottom_index);
                atomicAdd(offset_bottom_diff + d, diff_val);
            }
        }
    }
}

int BROI1DPoolBackwardLaucher(
    const float* top_diff, const float temporal_scale, const int batch_size, const int num_rois, const int depth,
    const int channels, const int pooled_depth, const int start_pooled_depth, const int end_pooled_depth, 
    const float bratio, const float* bottom_rois, float* bottom_diff, cudaStream_t stream)
{
    const int kThreadsPerBlock = 1024;
    const int output_size = batch_size * num_rois * (pooled_depth+start_pooled_depth+end_pooled_depth) * channels;
    cudaError_t err;

    BROI1DPoolBackward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
      output_size, top_diff, num_rois, temporal_scale, depth,
      channels, pooled_depth, start_pooled_depth, end_pooled_depth, 
      bratio, bottom_diff, bottom_rois);

    err = cudaGetLastError();
    if(cudaSuccess != err)
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}


#ifdef __cplusplus
}
#endif


