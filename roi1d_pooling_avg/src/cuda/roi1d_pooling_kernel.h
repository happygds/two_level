#ifndef _ROI1D_POOLING_KERNEL
#define _ROI1D_POOLING_KERNEL

#ifdef __cplusplus
extern "C" {
#endif

int ROI1DPoolForwardLaucher(
    const float* bottom_data, const float temporal_scale,
    const int num_rois, const int depth,
    const int channels, const int pooled_width,
    const float* bottom_rois, const int batch_size,
    float* top_data, cudaStream_t stream);


int ROI1DPoolBackwardLaucher(const float* top_diff, const float temporal_scale,
 	const int batch_size, const int num_rois, const int depth,
 	const int channels, const int pooled_depth,
 	const float* bottom_rois, float* bottom_diff, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif

