#include "tree_verify.h"
#include "tree_types.h"
#include "logsumexp.cuh"
#include "accept_chain.cuh"
#include "residual_sample.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t _err = (call); \
        if (_err != cudaSuccess) { \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(_err)); \
            abort(); \
        } \
    } while(0)

// CPU-side path enumeration.
int enumerate_paths(const int32_t* parent_ids,
                    const int32_t* depth_arr,
                    int            node_count,
                    int            path_nodes[MAX_TREE_WIDTH][MAX_TREE_DEPTH],
                    int            path_lengths[MAX_TREE_WIDTH])
{
    // Find leaves: nodes with no children.
    bool has_child[MAX_TREE_NODES] = {};
    for (int n = 0; n < node_count; ++n) {
        if (parent_ids[n] >= 0) has_child[parent_ids[n]] = true;
    }

    int num_paths = 0;
    for (int n = 0; n < node_count && num_paths < MAX_TREE_WIDTH; ++n) {
        if (has_child[n]) continue;  // not a leaf
        // Walk from leaf to root and reverse.
        int path[MAX_TREE_DEPTH];
        int len = 0;
        int cur = n;
        while (cur >= 0 && len < MAX_TREE_DEPTH) {
            path[len++] = cur;
            cur = parent_ids[cur];
        }
        // Reverse to get root-to-leaf order.
        path_lengths[num_paths] = len;
        for (int i = 0; i < len; ++i) {
            path_nodes[num_paths][i] = path[len - 1 - i];
        }
        ++num_paths;
    }
    return num_paths;
}

// Shared memory layout for tree verification.
// node_lse[node_id] holds the precomputed target log-sum-exp for each node.
// Sized for MAX_TREE_NODES; actual node_count may be smaller.

// Fused tree verification kernel.
// Grid: one block per batch element.
// Block: MAX_TREE_WIDTH warps = MAX_TREE_WIDTH * 32 threads.
//   Warp i handles path i in the tree.
// Phase 1: all warps cooperate to compute log-sum-exp for all tree nodes.
//   Each warp computes one node's target lse, looping through nodes strided by warp count.
// Phase 2: each warp runs the acceptance chain for its assigned path.
// Phase 3: after all warps report their accepted count via shared memory,
//   warp 0 identifies the longest accepted path and performs correction/bonus sampling.
__global__ void tree_verify_kernel(
        const TreeDraft*       __restrict__ inputs,
        TreeVerifyResult*      __restrict__ outputs,
        int                    node_count,
        int                    vocab_size,
        int                    num_paths,
        const int* __restrict__ path_nodes,   // [MAX_TREE_WIDTH * MAX_TREE_DEPTH]
        const int* __restrict__ path_lengths  // [MAX_TREE_WIDTH]
)
{
    extern __shared__ float smem[];

    const int bid      = blockIdx.x;
    const int lane     = threadIdx.x & 31;
    const int warp_id  = threadIdx.x >> 5;
    const int num_warps = blockDim.x >> 5;

    // Shared memory layout:
    //   float node_target_lse[MAX_TREE_NODES]
    //   float node_draft_lse[MAX_TREE_NODES]
    //   int   warp_accepted[MAX_TREE_WIDTH]
    //   int   warp_reject_node[MAX_TREE_WIDTH]
    float* node_target_lse = smem;
    float* node_draft_lse  = smem + MAX_TREE_NODES;
    int*   warp_accepted   = reinterpret_cast<int*>(smem + 2 * MAX_TREE_NODES);
    int*   warp_reject_node= warp_accepted + MAX_TREE_WIDTH;

    const TreeDraft& inp = inputs[bid];

    // Phase 1: compute log-sum-exp for each tree node.
    // Warp warp_id handles nodes: warp_id, warp_id + num_warps, ...
    for (int node = warp_id; node < node_count; node += num_warps) {
        float tlse = warp_log_sum_exp(inp.target_logits + (size_t)node * vocab_size, vocab_size);
        float dlse = warp_log_sum_exp(inp.draft_logits  + (size_t)node * vocab_size, vocab_size);
        if (lane == 0) {
            node_target_lse[node] = tlse;
            node_draft_lse[node]  = dlse;
        }
    }
    __syncthreads();

    // Phase 2: each warp runs the acceptance chain for its path.
    int my_accepted   = 0;
    int my_reject_node = -1;

    if (warp_id < num_paths) {
        const int* my_path    = path_nodes   + warp_id * MAX_TREE_DEPTH;
        int        path_len   = path_lengths[warp_id];

        // Build per-path per-position arrays from node data.
        // We walk the path and call run_accept_chain with inline lse arrays.
        // Since run_accept_chain expects contiguous arrays, we collect node-level
        // lse values and run a mini acceptance loop here.

        int accepted = 0;
        int rp = path_len;
        for (int pos = 0; pos < path_len; ++pos) {
            int node = my_path[pos];
            int tok  = inp.token_ids[node];
            float rv = inp.random_values[node];

            int accept = 0;
            if (lane == 0) {
                float tgt_logit = __half2float(
                        inp.target_logits[(size_t)node * vocab_size + tok]);
                float dft_logit = __half2float(
                        inp.draft_logits[(size_t)node * vocab_size + tok]);
                float log_p_tgt = tgt_logit - node_target_lse[node];
                float log_p_dft = dft_logit - node_draft_lse[node];

                if (log_p_dft < -87.0f || log_p_tgt < -87.0f) {
                    accept = 0;
                } else {
                    float log_ratio   = fminf(0.0f, log_p_tgt - log_p_dft);
                    float accept_prob = __expf(log_ratio);
                    accept = (rv < accept_prob) ? 1 : 0;
                }
            }
            accept = __shfl_sync(FULL_MASK, accept, 0);
            if (accept) {
                ++accepted;
            } else {
                rp = pos;
                break;
            }
        }
        my_accepted    = accepted;
        my_reject_node = (rp < path_len) ? my_path[rp] : -1;

        if (lane == 0) {
            warp_accepted[warp_id]    = accepted;
            warp_reject_node[warp_id] = my_reject_node;
        }
    }
    __syncthreads();

    // Phase 3: warp 0 finds the longest accepted path and performs sampling.
    if (warp_id == 0) {
        // Find best path.
        int best_path = 0;
        int best_len  = (lane == 0) ? warp_accepted[0] : 0;
        // All-lanes scan for max accepted length.
        for (int p = lane; p < num_paths; p += 32) {
            int a = warp_accepted[p];
            if (a > best_len) { best_len = a; best_path = p; }
        }
        // Warp-reduce to find global best.
        // Use shared memory trick: each lane writes its candidate.
        __shared__ int s_best_path_candidates[32];
        __shared__ int s_best_len_candidates[32];
        s_best_path_candidates[lane] = best_path;
        s_best_len_candidates[lane]  = best_len;
        __syncwarp();
        if (lane == 0) {
            int gb = 0, gl = 0;
            for (int l = 0; l < 32; ++l) {
                if (s_best_len_candidates[l] > gl) {
                    gl = s_best_len_candidates[l];
                    gb = s_best_path_candidates[l];
                }
            }
            // Store global best.
            s_best_path_candidates[0] = gb;
            s_best_len_candidates[0]  = gl;
        }
        __syncwarp();
        int gbest_path = s_best_path_candidates[0];
        int gbest_len  = s_best_len_candidates[0];

        if (lane == 0) {
            TreeVerifyResult& out = outputs[bid];
            out.path_length = gbest_len;

            const int* best_p = path_nodes + gbest_path * MAX_TREE_DEPTH;
            int best_path_len = path_lengths[gbest_path];
            for (int i = 0; i < gbest_len; ++i) {
                int node = best_p[i];
                out.path_tokens[i] = inp.token_ids[node];
                out.path_log_probs[i] = __half2float(
                        inp.target_logits[(size_t)node * vocab_size + inp.token_ids[node]])
                        - node_target_lse[node];
            }

            // Correction or bonus sampling.
            int rn = warp_reject_node[gbest_path];
            float rv = 0.5f;
            int sampled_tok = 0;
            float sampled_lp = 0.0f;

            if (rn >= 0) {
                // Correction from residual at rejection node.
                rv = inp.random_values[rn];
                sampled_tok = gbest_len;  // placeholder; actual call needs warp
                // Store for post-sync warp call (we defer to after __syncthreads).
                // For now, store the rejection node index.
                out.path_tokens[gbest_len]   = -rn - 1;  // sentinel: negative = need sampling
                out.path_log_probs[gbest_len] = (float)rn;
            } else {
                // Bonus token from target distribution at position k.
                // Use last node on best path as the "k+1" position.
                int last_node = best_p[best_path_len - 1];
                rv = inp.random_values[last_node];
                out.path_tokens[gbest_len]   = -last_node - 2;  // sentinel
                out.path_log_probs[gbest_len] = rv;
            }
            (void)sampled_tok; (void)sampled_lp;
        }
        __syncwarp();

        // Now do the actual sampling with all 32 lanes of warp 0.
        // Read back sentinel from shared output.
        __shared__ int s_best_path_final, s_best_len_final;
        if (lane == 0) {
            s_best_path_final = s_best_path_candidates[0];
            s_best_len_final  = s_best_len_candidates[0];
        }
        __syncwarp();

        int gbp = s_best_path_final;
        int gbl = s_best_len_final;
        int rn  = warp_reject_node[gbp];

        int sampled_tok;
        float sampled_lp;

        if (rn >= 0) {
            float tlse = node_target_lse[rn];
            float dlse = node_draft_lse[rn];
            float rv   = inp.random_values[rn];
            sampled_tok = warp_correction_sample(
                    inp.target_logits + (size_t)rn * vocab_size,
                    inp.draft_logits  + (size_t)rn * vocab_size,
                    tlse, dlse, rv, vocab_size);
            if (lane == 0) {
                sampled_lp = __half2float(
                        inp.target_logits[(size_t)rn * vocab_size + sampled_tok]) - tlse;
            }
        } else {
            // Bonus: sample from the target at the last node of the best path.
            const int* best_p = path_nodes + gbp * MAX_TREE_DEPTH;
            int best_path_len = path_lengths[gbp];
            int last_node = best_p[best_path_len - 1];
            float tlse = node_target_lse[last_node];
            float rv   = inp.random_values[last_node];
            sampled_tok = warp_bonus_sample(
                    inp.target_logits + (size_t)last_node * vocab_size,
                    tlse, rv, vocab_size);
            if (lane == 0) {
                sampled_lp = __half2float(
                        inp.target_logits[(size_t)last_node * vocab_size + sampled_tok]) - tlse;
            }
        }

        if (lane == 0) {
            outputs[bid].path_tokens[gbl]    = sampled_tok;
            outputs[bid].path_log_probs[gbl] = sampled_lp;
        }
    }
}

// Host function for tree verification.
void host_tree_verify(const TreeDraftConfig& cfg,
                       const TreeDraft*       d_inputs,
                       TreeVerifyResult*      d_outputs,
                       int                    batch_size,
                       const int*             d_path_nodes,    // [MAX_TREE_WIDTH * MAX_TREE_DEPTH]
                       const int*             d_path_lengths,  // [MAX_TREE_WIDTH]
                       int                    num_paths,
                       cudaStream_t           stream)
{
    const int V          = cfg.base.vocab_size;
    const int node_count = cfg.base.num_draft_tokens;

    // Shared memory:
    //   float node_target_lse[MAX_TREE_NODES]
    //   float node_draft_lse[MAX_TREE_NODES]
    //   int   warp_accepted[MAX_TREE_WIDTH]
    //   int   warp_reject_node[MAX_TREE_WIDTH]
    //   int   s_best_path_candidates[32]
    //   int   s_best_len_candidates[32]
    int smem_bytes = (2 * MAX_TREE_NODES + 2 * MAX_TREE_WIDTH + 64) * sizeof(int);

    // One warp per path; round up to max_tree_width warps.
    int threads_per_block = MAX_TREE_WIDTH * 32;
    (void)node_count;

    tree_verify_kernel<<<batch_size, threads_per_block, smem_bytes, stream>>>(
            d_inputs, d_outputs,
            cfg.base.num_draft_tokens,
            V, num_paths,
            d_path_nodes, d_path_lengths);
}
