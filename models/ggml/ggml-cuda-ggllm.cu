#if defined(GGML_USE_HIPBLAS)
#ifndef cudaMemGetInfo
#define cudaMemGetInfo hipMemGetInfo
#endif
#endif

// https://github.com/cmp-nct/ggllm.cpp/blob/master/ggml-cuda.cu

static GPUStatus g_system_gpu_status;

const GPUStatus* ggml_cuda_get_system_gpu_status(void) {
  return &g_system_gpu_status;
}

// Todo verify: free and total memory reported by cudaMemGetInfo differs from
// gpu_z which also differs from hwinfo64. Update the system status about
// available GPUs and memory usage
void ggml_cuda_update_gpu_status(int device_id) {
  int currentDevice = 0;
  CUDA_CHECK(cudaGetDevice(&currentDevice));
  if (device_id == -1) {
    // Update all devices
    if (g_system_gpu_status.num_devices == 0) {
      CUDA_CHECK(cudaGetDeviceCount(&g_system_gpu_status.num_devices));
      if (g_system_gpu_status.num_devices > GGML_CUDA_MAX_DEVICES) {
        g_system_gpu_status.num_devices = GGML_CUDA_MAX_DEVICES;
        fprintf(stderr,
                "WARNING: GGML_CUDA_MAX_DEVICES is smaller than the number of "
                "devices on the system. Using first %d devices.\n",
                GGML_CUDA_MAX_DEVICES);
      }
      if (g_system_gpu_status.max_gpus == 0) {
        g_system_gpu_status.max_gpus = 1;
      }
      if (g_system_gpu_status.num_devices > g_system_gpu_status.max_gpus)
        g_system_gpu_status.num_devices = g_system_gpu_status.max_gpus;

      g_system_gpu_status.total_vram = 0;
      for (int id = 0; id < g_system_gpu_status.num_devices; ++id) {
        CUDA_CHECK(
            cudaGetDeviceProperties(&g_system_gpu_status.device_props[id], id));
      }
    }
    g_system_gpu_status.total_vram = 0;
    g_system_gpu_status.total_free_vram = 0;
    for (int id = 0; id < g_system_gpu_status.num_devices; ++id) {
      CUDA_CHECK(cudaSetDevice(id));
      CUDA_CHECK(cudaMemGetInfo(&g_system_gpu_status.device_vram_free[id],
                                &g_system_gpu_status.device_vram_total[id]));
      g_system_gpu_status.total_vram +=
          g_system_gpu_status.device_vram_total[id];
      g_system_gpu_status.total_free_vram +=
          g_system_gpu_status.device_vram_free[id];
    }
    // restore current device
    if (currentDevice != g_system_gpu_status.num_devices - 1) {
      CUDA_CHECK(cudaSetDevice(currentDevice));
    }
  } else {
    // Update only the specified device
    CUDA_CHECK(cudaGetDeviceProperties(
        &g_system_gpu_status.device_props[device_id], device_id));
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(
        cudaMemGetInfo(&g_system_gpu_status.device_vram_free[device_id],
                       &g_system_gpu_status.device_vram_total[device_id]));
    // go through all devices and update total/free
    g_system_gpu_status.total_vram = 0;
    g_system_gpu_status.total_free_vram = 0;
    for (int id = 0; id < g_system_gpu_status.num_devices; ++id) {
      g_system_gpu_status.total_vram +=
          g_system_gpu_status.device_vram_total[id];
      g_system_gpu_status.total_free_vram +=
          g_system_gpu_status.device_vram_free[id];
    }
    // restore current device
    if (device_id != currentDevice) {
      CUDA_CHECK(cudaSetDevice(currentDevice));
    }
  }

#if 1
  // required for proper vram distribution but split tensors require memory on
  // primary GPU which could be disabled remove unused GPUs from available
  // calculation
  bool all_zero = true;
  for (int i = 0; i < g_system_gpu_status.num_devices; ++i) {
    if (g_tensor_split[i] != 0.0f) {
      all_zero = false;
    }
  }
  if (!all_zero)
    for (int id = 0; id < g_system_gpu_status.num_devices; ++id) {
      if (g_tensor_split[id] >= 1.0 ||
          (id > 0 && g_tensor_split[id] == g_tensor_split[id - 1])) {
        g_system_gpu_status.total_vram -=
            g_system_gpu_status.device_vram_total[id];
        g_system_gpu_status.total_free_vram -=
            g_system_gpu_status.device_vram_free[id];
      }
    }
#endif
}

// unallocates any "free" buffers that have not been used (or less than n times
// since last free) for example call after evaluation
int ggml_cuda_pool_purge_buffers_with_access_count(int min_access_count,
                                                   int device_id) {
  scoped_spin_lock lock(g_cuda_pool_lock);
  int id;
  CUDA_CHECK(cudaGetDevice(&id));

  int total_purged = 0;

  for (int i = 0; i < MAX_CUDA_BUFFERS; ++i) {
    cuda_buffer& b = g_cuda_buffer_pool[device_id][i];
    if (b.ptr != nullptr && b.access_count < min_access_count) {
      if (id != device_id) {
        CUDA_CHECK(cudaSetDevice(device_id));
      }
      CUDA_CHECK(cudaFree(b.ptr));
      // printf("\n-----> CUDA: access count - purged buffer %d of size %zu for
      // device %d\n", i, b.size, device_id);
      b.ptr = nullptr;
      b.size = 0;
      b.access_count = 0;

      total_purged++;
    }
  }
  return total_purged;
}

// resets access_count for all free buffers (for example before evaluation)
void ggml_cuda_pool_reset_all_counters(int device_id) {
  scoped_spin_lock lock(g_cuda_pool_lock);

  for (int i = 0; i < MAX_CUDA_BUFFERS; ++i) {
    cuda_buffer& b = g_cuda_buffer_pool[device_id][i];
    if (b.ptr != nullptr) {
      b.access_count = 0;
      // printf("CUDA: reset buffer %d of size %zu access_count %d for device
      // %d\n", i, b.size, b.access_count, device_id);
    }
  }
}
