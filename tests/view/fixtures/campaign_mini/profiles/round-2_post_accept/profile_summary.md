# Profile summary — round 2 (post_accept)

Top kernel families:

| family   | total ms | share |
|----------|----------|-------|
| fwd_dgrad | 3.40 | 49% |
| wgrad     | 2.05 | 30% |
| quant     | 0.62 | 9%  |
| amax      | 0.40 | 6%  |
| other     | 0.40 | 6%  |

Improvement: replacing the bf16→fp8 cast with `cast_only(scale=False)` cut the
forward gemm path by ~19%. VGPR pressure unchanged at 256.
