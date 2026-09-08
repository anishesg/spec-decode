#pragma once
#include "types.h"
#include <cuda_runtime.h>

// Launch the fused speculative verification kernel.
//
// One thread block per batch element. Each block uses 64 threads (2 warps):
//   - Warp 0: computes target log-sum-exp for positions 0..k.
//   - Warp 1: computes draft log-sum-exp for positions 0..k-1 (in parallel with warp 0).
//   - After sync: warp 0 runs the acceptance chain.
//   - If rejection: all warps cooperate on correction sampling.
//   - If all accepted: all warps cooperate on bonus token sampling.
//
// Shared memory layout (per block):
//   float target_lse[k+1]       // (k+1) * 4 bytes
//   float draft_lse[k]          // k * 4 bytes
//   int   draft_toks[k]         // k * 4 bytes  (copied from DraftResult)
//   float rand_vals[k+1]        // (k+1) * 4 bytes
//   (no CDF workspace needed; correction sampling is register-resident)
//
// host_fused_verify fills VerifyResult for each element in the batch.
void host_fused_verify(const SpecConfig&  cfg,
                        const DraftResult* d_inputs,  // device array [batch_size]
                        VerifyResult*      d_outputs, // device array [batch_size]
                        int                batch_size,
                        cudaStream_t       stream = 0);

// Returns the number of bytes of shared memory required per block.
int fused_verify_smem_bytes(const SpecConfig& cfg);
