#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <string>

// Forward declarations from CUDA source files.
#include "../src/types.h"
#include "../src/fused_verify.cuh"
#include "../src/reference.h"
#include "../src/tree_types.h"
#include "../src/tree_verify.h"

// Validate a logit tensor: must be CUDA, contiguous, float16 or float32.
static void validate_logit_tensor(const torch::Tensor& t,
                                   const std::string& name,
                                   int expected_rows, int expected_cols) {
    TORCH_CHECK(t.is_cuda(),       name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.dim() == 2,      name, " must be 2D [rows, vocab]");
    TORCH_CHECK(t.size(0) == expected_rows,
                name, " expected ", expected_rows, " rows, got ", t.size(0));
    TORCH_CHECK(t.size(1) == expected_cols,
                name, " expected vocab_size=", expected_cols, " cols, got ", t.size(1));
    TORCH_CHECK(t.scalar_type() == torch::kFloat16 || t.scalar_type() == torch::kFloat32,
                name, " must be float16 or float32");
}

// Convert a float32 logit tensor to float16 if needed (returns existing tensor if already fp16).
static torch::Tensor to_fp16(const torch::Tensor& t) {
    if (t.scalar_type() == torch::kFloat16) return t;
    return t.to(torch::kFloat16);
}

// fused_verify: single-kernel speculative verification for a batch.
// Inputs:
//   target_logits: [batch, k+1, V] float16/float32
//   draft_logits:  [batch, k, V]   float16/float32
//   draft_tokens:  [batch, k]      int64
//   random_values: [batch, k+1]    float32
// Returns:
//   dict with keys: num_accepted [batch], tokens [batch, k+1], log_probs [batch, k+1]
py::dict fused_verify_batch(
        torch::Tensor target_logits,
        torch::Tensor draft_logits,
        torch::Tensor draft_tokens,
        torch::Tensor random_values)
{
    TORCH_CHECK(target_logits.dim() == 3, "target_logits must be [batch, k+1, V]");
    int batch = (int)target_logits.size(0);
    int k     = (int)draft_logits.size(1);
    int V     = (int)target_logits.size(2);

    TORCH_CHECK(target_logits.size(1) == k + 1, "target_logits must have k+1 rows per batch");
    TORCH_CHECK(draft_logits.size(0)  == batch, "batch size mismatch in draft_logits");
    TORCH_CHECK(draft_logits.size(1)  == k,     "draft_logits must have k rows per batch");
    TORCH_CHECK(draft_logits.size(2)  == V,     "vocab_size mismatch");
    TORCH_CHECK(draft_tokens.size(0)  == batch && draft_tokens.size(1) == k,
                "draft_tokens must be [batch, k]");
    TORCH_CHECK(random_values.size(0) == batch && random_values.size(1) == k+1,
                "random_values must be [batch, k+1]");

    TORCH_CHECK(target_logits.is_cuda() && draft_logits.is_cuda() &&
                draft_tokens.is_cuda() && random_values.is_cuda(),
                "all tensors must be on CUDA");

    auto tgt_fp16  = to_fp16(target_logits.contiguous());
    auto dft_fp16  = to_fp16(draft_logits.contiguous());
    auto toks_i32  = draft_tokens.to(torch::kInt32).contiguous();
    auto rv_f32    = random_values.contiguous();

    // Build per-element DraftResult structs on the host, then copy to device.
    std::vector<DraftResult> h_drs(batch);
    for (int b = 0; b < batch; ++b) {
        h_drs[b].target_logits = (__half*)tgt_fp16[b].data_ptr<at::Half>();
        h_drs[b].draft_logits  = (__half*)dft_fp16[b].data_ptr<at::Half>();
        h_drs[b].draft_tokens  = toks_i32[b].data_ptr<int32_t>();
        h_drs[b].random_values = rv_f32[b].data_ptr<float>();
    }
    auto d_drs_t = torch::from_blob(nullptr, {0});
    DraftResult* d_drs;
    cudaMalloc(&d_drs, batch * sizeof(DraftResult));
    cudaMemcpy(d_drs, h_drs.data(), batch * sizeof(DraftResult), cudaMemcpyHostToDevice);

    VerifyResult* d_out;
    cudaMalloc(&d_out, batch * sizeof(VerifyResult));

    SpecConfig cfg{V, k, k};
    host_fused_verify(cfg, d_drs, d_out, batch);
    cudaDeviceSynchronize();

    // Copy results to host.
    std::vector<VerifyResult> h_out(batch);
    cudaMemcpy(h_out.data(), d_out, batch * sizeof(VerifyResult), cudaMemcpyDeviceToHost);
    cudaFree(d_drs);
    cudaFree(d_out);

    // Pack into torch tensors.
    auto num_accepted_t = torch::empty({batch}, torch::kInt32);
    auto tokens_t       = torch::empty({batch, k+1}, torch::kInt32);
    auto log_probs_t    = torch::empty({batch, k+1}, torch::kFloat32);
    for (int b = 0; b < batch; ++b) {
        num_accepted_t[b] = h_out[b].num_accepted;
        for (int i = 0; i <= k; ++i) {
            tokens_t[b][i]    = h_out[b].output_tokens[i];
            log_probs_t[b][i] = h_out[b].log_probs[i];
        }
    }

    py::dict result;
    result["num_accepted"] = num_accepted_t;
    result["tokens"]       = tokens_t;
    result["log_probs"]    = log_probs_t;
    return result;
}

// Single-instance wrapper matching the Python API.
py::dict fused_verify_single(
        torch::Tensor target_logits,  // [k+1, V]
        torch::Tensor draft_logits,   // [k, V]
        torch::Tensor draft_tokens,   // [k]
        torch::Tensor random_values)  // [k+1]
{
    return fused_verify_batch(
            target_logits.unsqueeze(0),
            draft_logits.unsqueeze(0),
            draft_tokens.unsqueeze(0),
            random_values.unsqueeze(0));
}

// reference_verify: CPU sequential reference implementation.
py::dict reference_verify(
        torch::Tensor target_logits,  // [k+1, V] float16/float32, CPU tensor
        torch::Tensor draft_logits,   // [k, V]   float16/float32, CPU tensor
        torch::Tensor draft_tokens,   // [k]      int64, CPU tensor
        torch::Tensor random_values)  // [k+1]    float32, CPU tensor
{
    TORCH_CHECK(!target_logits.is_cuda(), "reference_verify expects CPU tensors");
    int k = (int)draft_logits.size(0);
    int V = (int)target_logits.size(1);

    auto tgt_fp16 = to_fp16(target_logits.contiguous());
    auto dft_fp16 = to_fp16(draft_logits.contiguous());
    auto toks_i32 = draft_tokens.to(torch::kInt32).contiguous();
    auto rv_f32   = random_values.contiguous();

    SpecConfig cfg{V, k, k};
    VerifyResult res = verify_reference(cfg, DraftResult{},
                                         (__half*)tgt_fp16.data_ptr<at::Half>(),
                                         (__half*)dft_fp16.data_ptr<at::Half>(),
                                         toks_i32.data_ptr<int32_t>(),
                                         rv_f32.data_ptr<float>());

    py::dict result;
    result["num_accepted"] = py::int_(res.num_accepted);
    auto tokens_t    = torch::empty({k+1}, torch::kInt32);
    auto log_probs_t = torch::empty({k+1}, torch::kFloat32);
    for (int i = 0; i <= k; ++i) {
        tokens_t[i]    = res.output_tokens[i];
        log_probs_t[i] = res.log_probs[i];
    }
    result["tokens"]    = tokens_t;
    result["log_probs"] = log_probs_t;
    return result;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Fused GPU speculative decoding verification";

    m.def("fused_verify_single", &fused_verify_single,
          "Single-instance fused verification",
          py::arg("target_logits"),
          py::arg("draft_logits"),
          py::arg("draft_tokens"),
          py::arg("random_values"));

    m.def("fused_verify_batch", &fused_verify_batch,
          "Batched fused verification",
          py::arg("target_logits"),
          py::arg("draft_logits"),
          py::arg("draft_tokens"),
          py::arg("random_values"));

    m.def("reference_verify", &reference_verify,
          "CPU sequential reference verification",
          py::arg("target_logits"),
          py::arg("draft_logits"),
          py::arg("draft_tokens"),
          py::arg("random_values"));
}
