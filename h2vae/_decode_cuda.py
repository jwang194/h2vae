"""CUDA bit-unpack + standardise kernel for the cohort cache.

Replaces the CPU-side ``decode_variant_chunk → astype(fp32) →
standardise → .to(device)`` pipeline.  Lets us transfer the
**bit-packed** chunk (130 MB at our scale) to GPU instead of the
decoded fp32 chunk (2 GB), and fold the unpack + standardise into one
GPU kernel.

Persistent cache stays on the host (CPU RAM); only one chunk's bit-
packed bytes ever land on the GPU.  This keeps VRAM usage independent
of (n_cohort × m_variants), letting modest-VRAM GPUs handle UKB-scale
heritability work.

Encoding (matches ``cohort_cache.CohortCache`` layout)::

    code 0b00 → genotype 0
    code 0b01 → genotype 1
    code 0b10 → genotype 2
    code 0b11 → missing (mean-imputed → 0 in standardised space)

The kernel reads one byte per thread (== 4 consecutive variants for
one sample) and writes 4 standardised fp32 outputs.  Block layout is
chosen so consecutive threads issue coalesced loads on ``packed`` and
coalesced stores on ``out``.
"""
from __future__ import annotations

import torch

# Lazy compile: import-time would force a CUDA build even when only
# using the CPU code path. Compile on first call.
_KERNEL = None


def _compile_kernel():
    """Build & load the CUDA extension.  Called once on first use."""
    from torch.utils.cpp_extension import load_inline

    cuda_src = r"""
    #include <cuda_runtime.h>
    #include <torch/extension.h>
    #include <stdint.h>

    __global__ void decode_and_standardise_kernel(
        const uint8_t* __restrict__ packed,    // (n × byte_count) row-major
        int n, int byte_count, int chunk_var,
        const float* __restrict__ mean,        // (chunk_var,)
        const float* __restrict__ sd,          // (chunk_var,)
        float* __restrict__ out                // (n × chunk_var) row-major
    ) {
        int j_byte = blockIdx.x * blockDim.x + threadIdx.x;
        if (j_byte >= byte_count) return;
        int j_base = j_byte * 4;

        // Grid-stride loop on sample axis: gridDim.y is capped at 65535
        // (CUDA limit), so each block iterates if n is larger.
        for (int i = blockIdx.y; i < n; i += gridDim.y) {
            uint8_t b = packed[i * byte_count + j_byte];
            float* out_row = out + i * chunk_var;

            #pragma unroll
            for (int l = 0; l < 4; l++) {
                int j = j_base + l;
                if (j >= chunk_var) break;
                int code = (b >> (l * 2)) & 0x3;
                float v;
                if (code == 3) {
                    // missing → mean-imputed → 0 in standardised space
                    v = 0.0f;
                } else {
                    v = (static_cast<float>(code) - mean[j]) / sd[j];
                }
                out_row[j] = v;
            }
        }
    }

    void decode_and_standardise(
        torch::Tensor packed,        // (n, byte_count) uint8 CUDA
        torch::Tensor mean,          // (chunk_var,) fp32 CUDA
        torch::Tensor sd,            // (chunk_var,) fp32 CUDA
        torch::Tensor out            // (n, chunk_var) fp32 CUDA
    ) {
        TORCH_CHECK(packed.is_cuda(), "packed must be CUDA");
        TORCH_CHECK(mean  .is_cuda(), "mean must be CUDA");
        TORCH_CHECK(sd    .is_cuda(), "sd must be CUDA");
        TORCH_CHECK(out   .is_cuda(), "out must be CUDA");
        TORCH_CHECK(packed.dtype() == torch::kUInt8,   "packed must be uint8");
        TORCH_CHECK(mean  .dtype() == torch::kFloat32, "mean must be fp32");
        TORCH_CHECK(sd    .dtype() == torch::kFloat32, "sd must be fp32");
        TORCH_CHECK(out   .dtype() == torch::kFloat32, "out must be fp32");
        TORCH_CHECK(packed.is_contiguous() && mean.is_contiguous()
                    && sd.is_contiguous() && out.is_contiguous(),
                    "tensors must be contiguous");

        int n          = packed.size(0);
        int byte_count = packed.size(1);
        int chunk_var  = out.size(1);
        TORCH_CHECK(out.size(0) == n, "out.size(0) must match packed.size(0)");
        TORCH_CHECK(mean.size(0) == chunk_var, "mean.size(0) must equal chunk_var");
        TORCH_CHECK(sd.size(0) == chunk_var,   "sd.size(0) must equal chunk_var");

        const int threads = 256;
        const int max_grid_y = 65535;
        int grid_y = n < max_grid_y ? n : max_grid_y;
        dim3 block(threads);
        dim3 grid((byte_count + threads - 1) / threads, grid_y);

        decode_and_standardise_kernel<<<grid, block>>>(
            packed.data_ptr<uint8_t>(),
            n, byte_count, chunk_var,
            mean.data_ptr<float>(),
            sd.data_ptr<float>(),
            out.data_ptr<float>()
        );
        cudaError_t err = cudaGetLastError();
        TORCH_CHECK(err == cudaSuccess, "kernel launch failed: ",
                    cudaGetErrorString(err));
    }
    """

    cpp_src = r"""
    void decode_and_standardise(
        torch::Tensor packed,
        torch::Tensor mean,
        torch::Tensor sd,
        torch::Tensor out
    );
    """

    return load_inline(
        name="h2vae_decode_cuda",
        cpp_sources=cpp_src,
        cuda_sources=cuda_src,
        functions=["decode_and_standardise"],
        verbose=False,
    )


def decode_and_standardise(
    packed: torch.Tensor,
    mean: torch.Tensor,
    sd: torch.Tensor,
    chunk_var: int | None = None,
) -> torch.Tensor:
    """Bit-unpack a sample-major chunk + standardise into fp32.

    Args:
        packed: ``(n_cohort, byte_count)`` uint8 CUDA tensor, slice of
            the host bit-packed cache that's been transferred to GPU.
        mean: ``(chunk_var,)`` fp32 CUDA tensor (per-variant means for
            this chunk).
        sd: ``(chunk_var,)`` fp32 CUDA tensor (per-variant std devs).
        chunk_var: number of variants in this chunk.  Defaults to
            ``byte_count * 4`` (only differs for the trailing partial
            chunk when ``m % 4 != 0``).

    Returns:
        ``(n_cohort, chunk_var)`` fp32 CUDA tensor of standardised
        genotypes with missing entries set to 0 (i.e., mean-imputed).
    """
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _compile_kernel()
    if chunk_var is None:
        chunk_var = packed.size(1) * 4
    out = torch.empty((packed.size(0), chunk_var),
                      dtype=torch.float32, device=packed.device)
    _KERNEL.decode_and_standardise(packed, mean, sd, out)
    return out
