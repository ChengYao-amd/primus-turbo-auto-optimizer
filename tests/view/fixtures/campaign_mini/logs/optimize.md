# Optimization Log

## Baseline

Baseline persistent kernel reached 1309.062 TFLOPS step geomean across
the 5 representative shapes. Forward avg 1308.688, backward avg 1309.436.

## Optimization History

### round-2 — ACCEPTED

Tightened `waves_per_eu` from 0 to 2 on `_grouped_fp8_persistent_gemm_kernel`.
Step geomean +0.448% (1309.062 -> 1314.927).

### round-3 — ROLLBACK

Applied same launch knob to backward kernel. No metric improved over
current best.

## Current Best

round-2 — step geomean 1314.927 TFLOPS (+0.448% vs baseline).

## Directions to Try

(none yet — ANALYZE in flight)

## Verified Ineffective Directions

- waves_per_eu on backward kernel (round-3)

## Final Report

(pending)
