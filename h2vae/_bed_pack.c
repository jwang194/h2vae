/* Fused BED → cohort-cache + variant-stats kernel.
 *
 * Per chunk:
 *   - reads BED bytes for a contiguous variant range,
 *   - for each cohort sample, extracts its 2-bit code,
 *   - packs into the bit-packed sample-major cache (4 variants per byte),
 *   - accumulates per-variant sum, sum-of-squares, and observed-count
 *     for the stats stream.
 *
 * Replaces the numpy intermediate (n_var × n_total) int8 matrix plus
 * the (n_var × n_cohort) fancy-index copy plus the separate pack-and-
 * stats passes. Equivalent in shape to SCORE's read_bed_mailman_stream
 * inner loop (SCORE/genotype.cpp:406).
 *
 * BED 2-bit code interpretation (PLINK1, allele-1 count):
 *   00 → 2  (homozygous A1)
 *   01 → -1 (missing)
 *   10 → 1  (heterozygous)
 *   11 → 0  (homozygous A2)
 *
 * Cache 2-bit code (matches CohortCache in cohort_cache.py):
 *   00 → 0    01 → 1    10 → 2    11 → missing
 */
#include <stdint.h>

static const uint8_t BED_TO_CACHE[4] = {2, 3, 1, 0};
static const int8_t  BED_TO_GENO [4] = {2, -1, 1, 0};

/* Variant-major int8 cache decoder.
 *
 * Writes a contiguous (n_var × n_cohort) int8 buffer with cohort
 * genotypes for the given BED block. Each variant's row is written
 * sequentially — the inner loop produces n_cohort sequential bytes
 * per outer iteration, which is bandwidth-bound rather than TLB-bound.
 *
 * Output encoding (int8):
 *   2  → homozygous A1
 *   1  → heterozygous
 *   0  → homozygous A2
 *   -1 → missing
 *
 * Also accumulates per-variant int64 stats.
 */
/* Transpose-and-pack a variant-major int8 buffer (m × n_cohort)
 * into a sample-major bit-packed cache (n_cohort × bytes_per_sample).
 *
 * The variant-inner loop reads 4 cache-friendly rows of X_var at a
 * time (510 KB span for n_cohort=127k) and emits one packed byte per
 * sample per group of 4 variants. Cache_out writes are strided across
 * sample rows but each cache line is reused 64 times within a span of
 * 64 consecutive j_byte values — so the effective write cost is
 * memory-bandwidth-bound rather than DRAM-latency-bound.
 *
 * Encoding (matches CohortCache):
 *   int8 0 → 0b00,  1 → 0b01,  2 → 0b10,  -1 → 0b11
 */
#ifndef TRANSPOSE_BS
#define TRANSPOSE_BS 1024      /* sample-block size */
#endif
#ifndef TRANSPOSE_BJ
#define TRANSPOSE_BJ 64        /* j_byte-block size (1 cache line worth) */
#endif

void transpose_int8_to_bitpacked(
    const int8_t* X_var,             /* (m × n_cohort) row-major */
    int64_t       m,
    int64_t       n_cohort,
    uint8_t*      cache_out,         /* (n_cohort × bytes_per_sample) row-major */
    int64_t       bytes_per_sample   /* = (m + 3) / 4 */
) {
    int64_t m4 = m / 4;
    int64_t m_rem = m - m4 * 4;

    /* Tiled transpose: outer over (sample_block, j_byte_block) so that
     * the touched cache rows fit in L2 and the inner loop's writes are
     * sequential within each sample's cache row. */
    for (int64_t i_blk = 0; i_blk < n_cohort; i_blk += TRANSPOSE_BS) {
        int64_t i_end = i_blk + TRANSPOSE_BS;
        if (i_end > n_cohort) i_end = n_cohort;
        for (int64_t jb_blk = 0; jb_blk < m4; jb_blk += TRANSPOSE_BJ) {
            int64_t jb_end = jb_blk + TRANSPOSE_BJ;
            if (jb_end > m4) jb_end = m4;
            for (int64_t i = i_blk; i < i_end; i++) {
                uint8_t* dst_row = cache_out + i * bytes_per_sample;
                for (int64_t j_byte = jb_blk; j_byte < jb_end; j_byte++) {
                    int8_t v0 = X_var[(j_byte * 4 + 0) * n_cohort + i];
                    int8_t v1 = X_var[(j_byte * 4 + 1) * n_cohort + i];
                    int8_t v2 = X_var[(j_byte * 4 + 2) * n_cohort + i];
                    int8_t v3 = X_var[(j_byte * 4 + 3) * n_cohort + i];
                    uint8_t c0 = (v0 == -1) ? 3 : (uint8_t)v0;
                    uint8_t c1 = (v1 == -1) ? 3 : (uint8_t)v1;
                    uint8_t c2 = (v2 == -1) ? 3 : (uint8_t)v2;
                    uint8_t c3 = (v3 == -1) ? 3 : (uint8_t)v3;
                    dst_row[j_byte] =
                        c0 | (c1 << 2) | (c2 << 4) | (c3 << 6);
                }
            }
        }
    }
    /* Trailing partial group (m % 4 != 0): pad with missing codes. */
    if (m_rem > 0) {
        for (int64_t i = 0; i < n_cohort; i++) {
            uint8_t b = 0;
            for (int l = 0; l < (int)m_rem; l++) {
                int8_t v = X_var[(m4 * 4 + l) * n_cohort + i];
                uint8_t c = (v == -1) ? 3 : (uint8_t)v;
                b |= c << (l * 2);
            }
            for (int l = (int)m_rem; l < 4; l++) {
                b |= 3 << (l * 2);
            }
            cache_out[i * bytes_per_sample + m4] = b;
        }
    }
}


void bed_decode_to_variant_int8(
    const uint8_t* bed_block,
    int64_t        n_var,
    int64_t        bytes_per_var_BED,
    const int64_t* row_idx,
    int64_t        n_cohort,
    int8_t*        out_int8,         /* (n_var × n_cohort) row-major */
    int64_t*       sum_x,            /* (n_var,) */
    int64_t*       sum_x2,
    int64_t*       n_obs
) {
    for (int64_t v = 0; v < n_var; v++) {
        const uint8_t* variant_bytes = bed_block + v * bytes_per_var_BED;
        int8_t* out_row = out_int8 + v * n_cohort;

        int64_t s_sum_x  = 0;
        int64_t s_sum_x2 = 0;
        int64_t s_n_obs  = 0;

        for (int64_t i = 0; i < n_cohort; i++) {
            int64_t bed_row = row_idx[i];
            uint8_t byte    = variant_bytes[bed_row >> 2];
            int     shift   = (int)((bed_row & 3) << 1);
            int     code    = (byte >> shift) & 0x3;
            int8_t  geno    = BED_TO_GENO[code];
            out_row[i] = geno;
            if (geno >= 0) {
                s_sum_x  += geno;
                s_sum_x2 += (int64_t)geno * (int64_t)geno;
                s_n_obs  += 1;
            }
        }

        sum_x [v] = s_sum_x;
        sum_x2[v] = s_sum_x2;
        n_obs [v] = s_n_obs;
    }
}


/* Sample-tiled variant: process SAMPLE_TILE samples at a time and sweep
 * variants inside that tile. This keeps the touched cache rows in L2/L3
 * across many variant updates, drastically reducing the per-cache-line
 * cost compared to the straightforward variant-outer / sample-inner
 * loop (which hits all 127k cache rows per variant and is TLB-bound).
 *
 * Stats are accumulated into the per-variant counters across all tiles.
 */
#ifndef SAMPLE_TILE
#define SAMPLE_TILE 4096
#endif

void bed_decode_and_pack(
    const uint8_t* bed_block,        /* (n_var × bytes_per_var_BED) view of the BED mmap */
    int64_t        n_var,
    int64_t        bytes_per_var_BED,
    const int64_t* row_idx,          /* (n_cohort,) BED row indices */
    int64_t        n_cohort,
    uint8_t*       cache_buf,        /* (n_cohort × bytes_per_sample) sample-major cache */
    int64_t        bytes_per_sample,
    int64_t        j_lo,             /* must be a multiple of 4 */
    int64_t*       sum_x,            /* (n_var,) int64 accumulators */
    int64_t*       sum_x2,
    int64_t*       n_obs
) {
    /* Zero accumulators first; later tile passes accumulate additively. */
    for (int64_t v = 0; v < n_var; v++) {
        sum_x [v] = 0;
        sum_x2[v] = 0;
        n_obs [v] = 0;
    }

    for (int64_t i_blk = 0; i_blk < n_cohort; i_blk += SAMPLE_TILE) {
        int64_t i_end = i_blk + SAMPLE_TILE;
        if (i_end > n_cohort) i_end = n_cohort;

        for (int64_t v = 0; v < n_var; v++) {
            const uint8_t* variant_bytes = bed_block + v * bytes_per_var_BED;
            int64_t j_global       = j_lo + v;
            int64_t cache_byte_idx = j_global >> 2;
            int     cache_bit_pos  = (int)((j_global & 3) << 1);

            int64_t s_sum_x  = 0;
            int64_t s_sum_x2 = 0;
            int64_t s_n_obs  = 0;

            for (int64_t i = i_blk; i < i_end; i++) {
                int64_t bed_row = row_idx[i];
                uint8_t byte    = variant_bytes[bed_row >> 2];
                int     shift   = (int)((bed_row & 3) << 1);
                int     code    = (byte >> shift) & 0x3;
                int     cache_code = BED_TO_CACHE[code];
                int8_t  geno       = BED_TO_GENO [code];

                cache_buf[i * bytes_per_sample + cache_byte_idx] |=
                    (uint8_t)(cache_code << cache_bit_pos);

                if (geno >= 0) {
                    s_sum_x  += geno;
                    s_sum_x2 += (int64_t)geno * (int64_t)geno;
                    s_n_obs  += 1;
                }
            }

            sum_x [v] += s_sum_x;
            sum_x2[v] += s_sum_x2;
            n_obs [v] += s_n_obs;
        }
    }
}
