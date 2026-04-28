# Profile summary — round 1 (baseline)

Top kernel families:

| family   | total ms | share |
|----------|----------|-------|
| fwd_dgrad | 4.20 | 53% |
| wgrad     | 2.10 | 27% |
| quant     | 0.80 | 10% |
| amax      | 0.45 | 6%  |
| other     | 0.32 | 4%  |

Bottleneck: `_grouped_fp8_persistent_gemm_kernel` (3.10 ms) is the dominant
forward kernel. VGPR pressure (256) exceeds occupancy threshold; consider
reducing accum-VGPR usage or reordering fp8 quant-amax fusion.
