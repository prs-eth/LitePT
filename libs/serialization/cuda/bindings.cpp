#include <torch/extension.h>

torch::Tensor hilbert_encode_cuda(torch::Tensor coords, int num_bits);

torch::Tensor morton_encode_cuda(torch::Tensor coords);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hilbert_encode", &hilbert_encode_cuda, "Hilbert encode (CUDA)");

    // morton (or Z-order) encoding
    m.def("morton_encode", &morton_encode_cuda, "Morton encode (CUDA)");
}