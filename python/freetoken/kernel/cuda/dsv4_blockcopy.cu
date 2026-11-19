// Batched block copy between device memory and pinned host memory (UVA: a pinned host pointer is
// directly addressable from the GPU), used by the DSV4 host L2 prefix cache (kvcache/dsv4_l2.py).
// Copies n blocks of `block_bytes` each: dst + dst_off[i]  <-  src + src_off[i]. Offsets are byte
// offsets (int64, device arrays); block_bytes and all offsets must be multiples of 16.
#include <cuda_runtime.h>
#include <stdint.h>

__global__ void block_copy_kernel(const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
                                  const int64_t* __restrict__ src_off, const int64_t* __restrict__ dst_off,
                                  long block_bytes) {
  const long i = blockIdx.x;
  const uint4* s = reinterpret_cast<const uint4*>(src + src_off[i]);
  uint4* d = reinterpret_cast<uint4*>(dst + dst_off[i]);
  const long n16 = block_bytes >> 4;
  for (long j = (long)blockIdx.y * blockDim.x + threadIdx.x; j < n16; j += (long)gridDim.y * blockDim.x) d[j] = s[j];
}

extern "C" int dsv4_block_copy(const void* src, void* dst, const void* src_off, const void* dst_off, long n,
                               long block_bytes, void* stream) {
  if (n == 0 || block_bytes == 0) return 0;
  if ((block_bytes & 15) != 0) return -1;
  const long n16 = block_bytes >> 4;
  int ychunks = (int)((n16 + 256 * 8 - 1) / (256 * 8));
  if (ychunks < 1) ychunks = 1;
  if (ychunks > 64) ychunks = 64;
  for (long b0 = 0; b0 < n; b0 += 65535) {
    const long nb = (n - b0) < 65535 ? (n - b0) : 65535;
    block_copy_kernel<<<dim3((unsigned)nb, ychunks), 256, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)src, (uint8_t*)dst, (const int64_t*)src_off + b0, (const int64_t*)dst_off + b0, block_bytes);
  }
  return (int)cudaGetLastError();
}
