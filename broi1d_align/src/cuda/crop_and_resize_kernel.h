#ifndef _CropAndResize_Kernel
#define _CropAndResize_Kernel

#ifdef __cplusplus
extern "C" {
#endif

void CropAndResizeLaucher(
    const float* bottom_data, const float* bottom_rois,
    int num_rois, int batch_size, int channels, int depth,
    int crop_depth, const float temporal_scale, const int start_pooled_depth,
    const int end_pooled_depth, const float bratio,
    float extrapolation_value, float* top_data, cudaStream_t stream);

void CropAndResizeBackpropImageLaucher(
    const float* top_grad, const float* bottom_rois,
    int num_rois, int batch_size, int channels, int depth,
    int crop_depth, const float temporal_scale, const int start_pooled_depth,
    const int end_pooled_depth, const float bratio,
    float* bottom_grad, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
