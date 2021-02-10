kernel_forward_t = '''
#define CUDA_1D_KERNEL_LOOP(i, n)                                 \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

typedef float T;
extern "C"
__global__ void PSROIPoolForward(
    const int nthreads,
    const T* bottom_data,
    const T spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const T* bottom_rois,
    const int output_dim,
    const int group_size,
    T* top_data,
    int* mapping_channel) 
    {
        // index of current thread
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index >= nthreads)
        {
            return;
        }

        // The output is in order (n, ctop, ph, pw)
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int ctop = (index / pooled_width / pooled_height) % output_dim;
        int n = index / pooled_width / pooled_height / output_dim;

        // [start, end) interval for spatial sampling
        const T* offset_bottom_rois = bottom_rois + n * 5;
        int roi_batch_ind = offset_bottom_rois[0];
        T roi_start_w = static_cast<T>(
          roundf(offset_bottom_rois[1])) * spatial_scale;
        T roi_start_h = static_cast<T>(
          roundf(offset_bottom_rois[2])) * spatial_scale;
        T roi_end_w = static_cast<T>(
          roundf(offset_bottom_rois[3]) + 1.) * spatial_scale;
        T roi_end_h = static_cast<T>(
          roundf(offset_bottom_rois[4]) + 1.) * spatial_scale;

        // Force too small ROIs to be 1x1
        T roi_width = max(roi_end_w - roi_start_w, static_cast<T>(0.1));  // avoid 0
        T roi_height = max(roi_end_h - roi_start_h, static_cast<T>(0.1));

        // Compute w and h at bottom
        T bin_size_h = roi_height / static_cast<T>(pooled_height);
        T bin_size_w = roi_width / static_cast<T>(pooled_width);

        // Add roi offsets and clip to input boundaries
        int hstart = floor(
          static_cast<T>(ph) * bin_size_h + roi_start_h);
        int wstart = floor(
          static_cast<T>(pw)* bin_size_w + roi_start_w);
        int hend = ceil(
          static_cast<T>(ph + 1) * bin_size_h + roi_start_h);
        int wend = ceil(
          static_cast<T>(pw + 1) * bin_size_w + roi_start_w);

        hstart = min(max(hstart, 0), height);
        hend = min(max(hend, 0), height);
        wstart = min(max(wstart, 0),width);
        wend = min(max(wend, 0), width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        /******************** Add sample step base on group_size ******************/
        /* the group of psROI pooling
         e.g. group_size=7, pooled_with=21, then module get 3x3 bottom data from each channel */
        
        // the horizontal index of the group of current pooling block
        int gw = floor(static_cast<T>(pw)* group_size / pooled_width);  
        // the vertical index of the group of current pooling block
        int gh = floor(static_cast<T>(ph)* group_size / pooled_height);

        // clip gw and gh to [0, group_size - 1]
        gw = min(max(gw, 0), group_size - 1);
        gh = min(max(gh, 0), group_size - 1);
        /********************                 end                ******************/

        int c = (ctop * group_size + gh) * group_size + gw;

        const T* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;

        T out_sum = 0;
        for (int h = hstart; h < hend; ++h){
         for (int w = wstart; w < wend; ++w){
           int bottom_index = h*width + w;
           out_sum += offset_bottom_data[bottom_index];
         }
        }

        T bin_area = (hend - hstart) * (wend - wstart);
        top_data[index] = is_empty ? 0. : out_sum / bin_area;
        mapping_channel[index] = c;
    }
'''

kernel_forward = '''
    #define CUDA_1D_KERNEL_LOOP(i, n)                               \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

    typedef float DType;
    extern "C"
    __global__ void PSROIPoolForwardKernel(
    const int count, 
    const DType* bottom_data, 
    const DType spatial_scale,
    const int channels, 
    const int height, 
    const int width,
    const int pooled_height,
    const int pooled_width, 
    const DType* bottom_rois, 
    const int output_dim, 
    const int group_size, 
    DType* top_data)
    {
        // get index of thread
        CUDA_1D_KERNEL_LOOP(index, count){

        // The output is in order (n, ctop, ph, pw)
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int ctop = (index / pooled_width / pooled_height) % output_dim;
        int n = index / pooled_width / pooled_height / output_dim;

        // [start, end) interval for spatial sampling
        const DType* offset_bottom_rois = bottom_rois + n * 5; 
        int roi_batch_ind = offset_bottom_rois[0];    // 该roi在batch中对应第几幅图, batch_idx
        DType roi_start_w = static_cast<DType>(round(offset_bottom_rois[1])) * spatial_scale;
        DType roi_start_h = static_cast<DType>(round(offset_bottom_rois[2])) * spatial_scale;
        DType roi_end_w = static_cast<DType>(round(offset_bottom_rois[3]) + 1.) * spatial_scale;
        DType roi_end_h = static_cast<DType>(round(offset_bottom_rois[4]) + 1.) * spatial_scale;
 
        DType roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0
        DType roi_height = max(roi_end_h - roi_start_h, 0.1);

        DType bin_size_h = roi_height / static_cast<DType>(pooled_height);
        DType bin_size_w = roi_width / static_cast<DType>(pooled_width);

        int hstart = floor(static_cast<DType>(ph) * bin_size_h + roi_start_h);
        int wstart = floor(static_cast<DType>(pw)* bin_size_w + roi_start_w);

        int hend = ceil(static_cast<DType>(ph + 1) * bin_size_h + roi_start_h);
        int wend = ceil(static_cast<DType>(pw + 1) * bin_size_w + roi_start_w);

        hstart = min(max(hstart, 0), height);
        hend = min(max(hend, 0), height);
        wstart = min(max(wstart, 0), width);
        wend = min(max(wend, 0), width);

        bool is_empty = (hend <= hstart) || (wend <= wstart);

        /* the group of psROI pooling
         e.g. group_size=7, pooled_with=21, then module get 3x3 bottom data from each channel */
        // the horizontal index of the group of current pooling block
        int gw = floor(static_cast<DType>(pw)* group_size / pooled_width); 
        // the vertical index of the group of current pooling block
        int gh = floor(static_cast<DType>(ph)* group_size / pooled_height);

        // clip gw and gh to [0, group_size - 1]
        gw = min(max(gw, 0), group_size - 1);
        gh = min(max(gh, 0), group_size - 1);

        // sample bottom data with Position-sensitive methods
        int c = (ctop*group_size + gh)*group_size + gw;

        const DType* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;

        DType out_sum = 0;
        for (int h = hstart; h < hend; ++h) {
          for (int w = wstart; w < wend; ++w) {
            int bottom_index = h*width + w;
            out_sum += offset_bottom_data[bottom_index];
          }
        }

        DType bin_area = (hend - hstart)*(wend - wstart);  
        top_data[index] = is_empty? (DType)0. : out_sum / bin_area; // avg pool 
        }   
    }
'''

kernel_backward_t = '''
    inline __device__
    float gpu_atomic_add(const float val, float* address) {
      return atomicAdd(address, val);
    }
typedef float T;
extern "C"
__global__ void PSROIPoolBackward(
    const int nthreads,
    const T* top_diff,
    const int* mapping_channel,
    const int num_rois,
    const T spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int output_dim,
    T* bottom_diff,
    const T* bottom_rois)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index >= nthreads)
        {
            return;
        }

        // The output is in order (n, ctop, ph, pw)
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int n = index / pooled_width / pooled_height / output_dim;

        // [start, end) interval for spatial sampling
        const T* offset_bottom_rois = bottom_rois + n * 5;    
        int roi_batch_ind = offset_bottom_rois[0];           
        T roi_start_w = static_cast<T>(roundf(offset_bottom_rois[1])) * spatial_scale;
        T roi_start_h = static_cast<T>(roundf(offset_bottom_rois[2])) * spatial_scale;
        T roi_end_w = static_cast<T>(roundf(offset_bottom_rois[3]) + 1.) * spatial_scale;
        T roi_end_h = static_cast<T>(roundf(offset_bottom_rois[4]) + 1.) * spatial_scale;

        // Force too small ROIs to be 1x1
        T roi_width = max(roi_end_w - roi_start_w, static_cast<T>(0.1)); //avoid 0
        T roi_height = max(roi_end_h - roi_start_h, static_cast<T>(0.1));

        T bin_size_h = roi_height / static_cast<T>(pooled_height);
        T bin_size_w = roi_width / static_cast<T>(pooled_width);

        int hstart = floor(static_cast<T>(ph)* bin_size_h + roi_start_h);
        int wstart = floor(static_cast<T>(pw)* bin_size_w + roi_start_w);
        
        int hend = ceil(static_cast<T>(ph + 1) * bin_size_h + roi_start_h);
        int wend = ceil(static_cast<T>(pw + 1) * bin_size_w + roi_start_w);

        hstart = min(max(hstart, 0), height);
        hend = min(max(hend, 0), height);
        wstart = min(max(wstart, 0), width);
        wend = min(max(wend, 0), width);

        bool is_empty = (hend <= hstart) || (wend <= wstart);

        int c = mapping_channel[index];
        T* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;
        T bin_area = (hend - hstart) * (wend - wstart);
        T diff_val = is_empty ? 0. : top_diff[index] / bin_area;
        for (int h = hstart; h < hend; ++h)
        {
          for (int w = wstart; w < wend; ++w)
          {
            int bottom_index = h * width + w;
            gpu_atomic_add(diff_val, offset_bottom_diff + bottom_index);
          }
        }
    }
'''

kernel_backward = '''
    typedef float DType;
    extern "C"
    __global__ void PSROIPoolBackwardAccKernel(
    const int count,
    const DType* top_diff,
    const int num_rois,
    const DType spatial_scale,
    const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int group_size,
    const int output_dim,
    DType* bottom_diff,
    const DType* bottom_rois)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index >= count)
        {
            return;
        }

        // The output is in order (n, ctop, ph, pw)
        int pw = index % pooled_width;
        int ph = (index / pooled_width) % pooled_height;
        int ctop = (index / pooled_width / pooled_height) % output_dim;
        int n = index / pooled_width / pooled_height / output_dim;

        // [start, end) interval for spatial sampling
        const DType* offset_bottom_rois = bottom_rois + n * 5;  
        int roi_batch_ind = offset_bottom_rois[0];              
        DType roi_start_w = static_cast<DType>(round(offset_bottom_rois[1])) * spatial_scale;
        DType roi_start_h = static_cast<DType>(round(offset_bottom_rois[2])) * spatial_scale;
        DType roi_end_w = static_cast<DType>(round(offset_bottom_rois[3]) + 1.) * spatial_scale;
        DType roi_end_h = static_cast<DType>(round(offset_bottom_rois[4]) + 1.) * spatial_scale;

        DType roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0
        DType roi_height = max(roi_end_h - roi_start_h, 0.1);

        DType bin_size_h = roi_height / static_cast<DType>(pooled_height);
        DType bin_size_w = roi_width / static_cast<DType>(pooled_width);

        int hstart = floor(static_cast<DType>(ph) * bin_size_h + roi_start_h);
        int wstart = floor(static_cast<DType>(pw)* bin_size_w + roi_start_w);

        int hend = ceil(static_cast<DType>(ph + 1) * bin_size_h + roi_start_h);
        int wend = ceil(static_cast<DType>(pw + 1) * bin_size_w + roi_start_w);

        hstart = min(max(hstart, 0), height);
        hend = min(max(hend, 0), height);
        wstart = min(max(wstart, 0), width);
        wend = min(max(wend, 0), width);

        bool is_empty = (hend <= hstart) || (wend <= wstart);

        /* the group of psROI pooling
         e.g. group_size=7, pooled_with=21, then module get 3x3 bottom data from each channel */
        int gw = floor(static_cast<DType>(pw)* group_size / pooled_width);  
        int gh = floor(static_cast<DType>(ph)* group_size / pooled_height); 

        gw = min(max(gw, 0), group_size - 1);
        gh = min(max(gh, 0), group_size - 1);

        int c = (ctop*group_size + gh)*group_size + gw;

        DType* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;

        DType bin_area = (hend - hstart)*(wend - wstart);    
        DType diff_val = is_empty ? (DType)0. : top_diff[index] / bin_area;

        // gradient backward
        for (int h = hstart; h < hend; ++h) 
        {
            for (int w = wstart; w < wend; ++w) 
            {
                int bottom_index = h*width + w;
                atomicAdd(offset_bottom_diff + bottom_index, diff_val);
            }
        }         
    }
'''



























