# Performance Trend

| Round | Status | Description | Fwd Avg TFLOPS | Fwd Peak TFLOPS | Bwd Avg TFLOPS | Bwd Peak TFLOPS | Step Geomean TFLOPS | vs Baseline | Key Finding |
|-------|--------|-------------|----------------|-----------------|----------------|-----------------|---------------------|-------------|-------------|
| 1 | BASELINE | Baseline persistent kernel | 1308.688 | 1777.602 | 1309.436 | 1767.153 | 1309.062 | — | starting point |
| 2 | ACCEPTED | Tighten waves_per_eu 0->2 on persistent forward | 1318.307 | 1808.410 | 1311.556 | 1767.587 | 1314.927 | step +0.448%, fwd +0.735%, bwd +0.162% | accept |
| 3 | ROLLBACK | Same launch knob on backward kernel | 1317.633 | 1807.462 | 1309.416 | 1767.796 | 1313.518 | step +0.340%, fwd +0.684%, bwd -0.002% | no metric improved |
