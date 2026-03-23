#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__device__ inline uint64_t split_by_3(uint32_t x) {
    uint64_t v = x & 0x1fffff; // up to 21 bits

    v = (v | (v << 32)) & 0x1f00000000ffff;
    v = (v | (v << 16)) & 0x1f0000ff0000ff;
    v = (v | (v << 8))  & 0x100f00f00f00f00f;
    v = (v | (v << 4))  & 0x10c30c30c30c30c3;
    v = (v | (v << 2))  & 0x1249249249249249;

    return v;
}

__device__ inline uint64_t morton3d_single(uint32_t x, uint32_t y, uint32_t z) {
    return (split_by_3(x) << 2) | (split_by_3(y) << 1) | split_by_3(z);
}

__global__ void morton_kernel(
    const uint32_t* __restrict__ coords,
    uint64_t* __restrict__ out,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    uint32_t x = coords[3 * idx + 0];
    uint32_t y = coords[3 * idx + 1];
    uint32_t z = coords[3 * idx + 2];

    out[idx] = morton3d_single(x, y, z);
}

torch::Tensor morton_encode_cuda(torch::Tensor coords) {
    TORCH_CHECK(coords.is_cuda(), "coords must be CUDA");
    TORCH_CHECK(coords.size(1) == 3, "coords must be (N,3)");

    int N = coords.size(0);

    auto coords_u32 = coords.to(torch::kUInt32).contiguous();
    auto out = torch::empty({N}, torch::dtype(torch::kUInt64).device(coords.device()));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    morton_kernel<<<blocks, threads>>>(
        coords_u32.data_ptr<uint32_t>(),
        out.data_ptr<uint64_t>(),
        N
    );

    return out;
}