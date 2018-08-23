int roi1d_pooling_forward_cuda(int pooled_depth, float temporal_scale, 
	THCudaTensor * features, THCudaTensor * rois, THCudaTensor * output);

int roi1d_pooling_backward_cuda(int pooled_depth, float temporal_scale, 
	THCudaTensor * top_grad, THCudaTensor * rois, THCudaTensor * bottom_grad);
