int broi1d_pooling_forward_cuda(int pooled_depth, float temporal_scale,
	int start_pooled_depth, int end_pooled_depth, float bratio,
	THCudaTensor * features, THCudaTensor * rois, THCudaTensor * output);

int broi1d_pooling_backward_cuda(int pooled_depth, float temporal_scale,
	int start_pooled_depth, int end_pooled_depth, float bratio,
	THCudaTensor * top_grad, THCudaTensor * rois, THCudaTensor * bottom_grad);
