#include <sanitizer_patching.h>

extern "C"
__device__ __noinline__
SanitizerPatchResult
sanitizer_memory_access_callback
(
 void* user_data,
 uint64_t pc,
 void* ptr,
 uint32_t size,
 uint32_t flags
) 
{
  int *data = (int *)user_data;
  atomicAdd(data, 1);
  return SANITIZER_PATCH_SUCCESS;
}


extern "C"
__device__ __noinline__
SanitizerPatchResult
sanitizer_barrier_callback
(
 void *user_data,
 uint64_t pc,
 uint32_t bar_index
)
{
  return SANITIZER_PATCH_SUCCESS;
}


extern "C"
__device__ __noinline__
SanitizerPatchResult
sanitizer_shfl_callback
(
 void *user_data,
 uint64_t pc
)
{
  return SANITIZER_PATCH_SUCCESS;
}


extern "C"
__device__ __noinline__
SanitizerPatchResult
sanitizer_block_enter_callback
(
 void *user_data
)
{
  int *data = (int *)user_data;
  atomicAdd(data, 1);
  return SANITIZER_PATCH_SUCCESS;
}


extern "C"
__device__ __noinline__
SanitizerPatchResult
sanitizer_block_exit_callback
(
 void *user_data,
 uint64_t pc
)
{
  return SANITIZER_PATCH_SUCCESS;
}
