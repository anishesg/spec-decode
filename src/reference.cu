#include "types.h"
#include "reference.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <stdexcept>

// CPU-side log-sum-exp over a half-precision logit row.
// Returns log(sum_v exp(logit[v])).
static double cpu_log_sum_exp(const __half* logits, int row, int vocab_size) {
    const __half* row_ptr = logits + (size_t)row * vocab_size;
    double max_val = -1e30;
    for (int v = 0; v < vocab_size; ++v) {
        double x = (double)__half2float(row_ptr[v]);
        if (x > max_val) max_val = x;
    }
    double sum = 0.0;
    for (int v = 0; v < vocab_size; ++v) {
        sum += std::exp((double)__half2float(row_ptr[v]) - max_val);
    }
    return max_val + std::log(sum);
}

// Sequential CPU reference implementation of speculative decoding verification.
// Correctness oracle; not performance-critical.
VerifyResult verify_reference(const SpecConfig& cfg,
                               const DraftResult& d,
                               const __half*  h_target_logits,
                               const __half*  h_draft_logits,
                               const int32_t* h_draft_tokens,
                               const float*   h_random_values) {
    const int k = cfg.num_draft_tokens;
    const int V = cfg.vocab_size;
    VerifyResult result{};

    // Pre-compute log-sum-exp (log partition function) for each row.
    std::vector<double> target_lse(k + 1), draft_lse(k);
    for (int pos = 0; pos <= k; ++pos) {
        target_lse[pos] = cpu_log_sum_exp(h_target_logits, pos, V);
    }
    for (int pos = 0; pos < k; ++pos) {
        draft_lse[pos] = cpu_log_sum_exp(h_draft_logits, pos, V);
    }

    // Sequential accept/reject loop.
    int accepted = 0;
    int rejection_pos = k;  // default: all accepted
    for (int pos = 0; pos < k; ++pos) {
        int tok = h_draft_tokens[pos];
        double log_p_target = (double)__half2float(h_target_logits[pos * V + tok]) - target_lse[pos];
        double log_p_draft  = (double)__half2float(h_draft_logits[pos * V + tok])  - draft_lse[pos];

        // Guard against -inf (zero probability) draft token.
        if (!std::isfinite(log_p_draft)) {
            rejection_pos = pos;
            break;
        }

        double log_ratio = std::min(0.0, log_p_target - log_p_draft);
        double accept_prob = std::exp(log_ratio);

        if ((double)h_random_values[pos] < accept_prob) {
            result.output_tokens[accepted] = tok;
            result.log_probs[accepted]     = (float)log_p_target;
            ++accepted;
        } else {
            rejection_pos = pos;
            break;
        }
    }

    result.num_accepted = accepted;

    // Correction sampling or bonus token sampling.
    int sample_pos = rejection_pos;  // position whose target distribution we sample from
    float rand_val = h_random_values[k];  // reuse last random value for correction/bonus

    if (rejection_pos < k) {
        // Sample from residual distribution: max(0, p_target[v] - p_draft[v]), normalized.
        double total_residual = 0.0;
        std::vector<double> residuals(V);
        for (int v = 0; v < V; ++v) {
            double p_t = std::exp((double)__half2float(h_target_logits[sample_pos * V + v]) - target_lse[sample_pos]);
            double p_d = std::exp((double)__half2float(h_draft_logits[sample_pos * V + v])  - draft_lse[sample_pos]);
            residuals[v] = std::max(0.0, p_t - p_d);
            total_residual += residuals[v];
        }
        double threshold = (double)rand_val * total_residual;
        double cumsum = 0.0;
        int sampled_tok = V - 1;
        for (int v = 0; v < V; ++v) {
            cumsum += residuals[v];
            if (cumsum > threshold) {
                sampled_tok = v;
                break;
            }
        }
        double log_p = (double)__half2float(h_target_logits[sample_pos * V + sampled_tok]) - target_lse[sample_pos];
        result.output_tokens[accepted] = sampled_tok;
        result.log_probs[accepted]     = (float)log_p;
    } else {
        // All k drafts accepted: sample bonus token from target[k].
        double threshold = (double)rand_val;
        double cumsum = 0.0;
        int sampled_tok = V - 1;
        for (int v = 0; v < V; ++v) {
            cumsum += std::exp((double)__half2float(h_target_logits[k * V + v]) - target_lse[k]);
            if (cumsum > threshold) {
                sampled_tok = v;
                break;
            }
        }
        double log_p = (double)__half2float(h_target_logits[k * V + sampled_tok]) - target_lse[k];
        result.output_tokens[accepted] = sampled_tok;
        result.log_probs[accepted]     = (float)log_p;
    }

    return result;
}
