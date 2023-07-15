// https://github.com/cmp-nct/ggllm.cpp/blob/master/ggml-cuda.h

typedef struct {
  int max_gpus;  // the max number of devices that can be used
  int num_devices;
  int main_device_id;
  size_t total_vram;
  size_t total_free_vram;
  size_t device_vram_free[GGML_CUDA_MAX_DEVICES];
  size_t device_vram_total[GGML_CUDA_MAX_DEVICES];
  int64_t
      device_vram_reserved[GGML_CUDA_MAX_DEVICES];  // overrides reserved vram -
                                                    // may be negative to force
                                                    // vram swapping
  struct cudaDeviceProp device_props[GGML_CUDA_MAX_DEVICES];

} GPUStatus;

const GPUStatus* ggml_cuda_get_system_gpu_status(void);
void ggml_cuda_update_gpu_status(int device_id);
void ggml_cuda_pool_reset_all_counters(int device_id);
int ggml_cuda_pool_purge_buffers_with_access_count(int min_access_count,
                                                   int device_id);
