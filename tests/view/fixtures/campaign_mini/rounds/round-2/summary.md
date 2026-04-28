# Round 2 — Set `waves_per_eu=2` on persistent forward

## Hypothesis

VGPR=128 places the wave at ~4 waves/EU on gfx950. Capping at 2 should
let the compiler hold more loop state in registers.

## Result

Step geomean **1314.927 TFLOPS** (+0.448% vs baseline).

## Decision

ACCEPTED.
