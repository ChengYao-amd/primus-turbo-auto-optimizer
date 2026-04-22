You are running the BASELINE phase (round-1) of the kernel-optimize loop.

Authoritative skill excerpt:

<skill>
{skill_excerpt}
</skill>

Manifest:

<manifest>
{manifest_yaml}
</manifest>

Campaign directory: `{campaign_dir}`
Round directory:    `{campaign_dir}/rounds/round-1/`
Structured output:  `{phase_result_path}`

{workspace_hygiene_block}
Tasks (strict order):

1. Run the focused test command from `manifest.test_command`. All tests
   MUST pass. If any fails, stop, record the failure reason in the JSON
   output (`test_pass=false`) and do NOT proceed to benchmarking.
2. Run the focused benchmark command from `manifest.benchmark_command`.
   Capture stdout / stderr to
   `{campaign_dir}/rounds/round-1/artifacts/benchmark.log`. If the
   project emits a CSV next to the current working directory, `mv` it
   to `{campaign_dir}/rounds/round-1/artifacts/benchmark.csv` (do NOT
   `cp` — see <workspace_hygiene>).
3. Parse the benchmark output per `manifest.primary_metric`. Compute the
   aggregate score (geometric mean of primary_metric across all PASS
   shapes) and a score vector (per-shape metrics).
4. Select 3-5 representative shapes covering small / medium / large
   behaviour from the PASS rows. These will drive `quick_command`.
5. Update the two files that must stay in sync with the chosen
   representative shapes:
   - `{campaign_dir}/quick_test_bench.py`: replace the empty `SHAPES`
     placeholder with the list you selected.
   - `{campaign_dir}/manifest.yaml`: set `representative_shapes` to the
     same list. Do not modify any other field unless explicitly required.
6. Run `manifest.quick_command` once against the freshly filled
   `representative_shapes`. Redirect the combined stdout+stderr to
   `{campaign_dir}/rounds/round-1/artifacts/quick_baseline.log`
   (append with `2>&1 | tee` or shell redirection; do NOT truncate
   after the fact). All shapes must PASS; if any check fails, stop
   the phase, emit `test_pass=false` in the JSON output, and leave
   `quick_baseline_log` set to the partial log path. Every later
   VALIDATE quick round will diff its own run against this reference.
7. Write `{campaign_dir}/rounds/round-1/summary.md` using the canonical
   round-summary template from the skill excerpt. Round-1's `Single
   change` block must explicitly say "No code change. Baseline round."
   and its `Decision` section must be `BASELINE`.
8. Copy the current kernel source into
   `{campaign_dir}/rounds/round-1/kernel_snapshot/` — note the trailing
   `/` so `cp` treats the destination as a directory (PREPARE_ENVIRONMENT
   may have done this already; verify and repair if missing).
9. Append a baseline block to `{campaign_dir}/logs/optimize.md` using the
   "Baseline record template" from the skill excerpt (do NOT rewrite any
   existing content; append only). The orchestrator will also append a
   `Quick baseline log:` line based on the `quick_baseline_log` field in
   the JSON output below, so you do not need to duplicate that line.
10. Append the first row to `{campaign_dir}/logs/performance_trend.md`
    using the Rule 8 table format (status `BASELINE`, vs baseline `—`).
11. Emit the structured phase result at `{phase_result_path}`:

```json
{{
  "test_pass": true_or_false,
  "benchmark_csv": "rounds/round-1/artifacts/benchmark.csv",
  "quick_baseline_log": "rounds/round-1/artifacts/quick_baseline.log",
  "primary_metric": "{primary_metric}",
  "aggregate_score": {{
    "Forward TFLOPS": <float>,
    "Backward TFLOPS": <float_or_null>
  }},
  "score_vector": [
    {{"shape": {{...}}, "check": "PASS", "metrics": {{
      "Forward TFLOPS": <float>,
      "Backward TFLOPS": <float_or_null>
    }}}}
  ],
  "trend_row": {{
    "fwd_avg":      <float>,
    "fwd_peak":     <float>,
    "bwd_avg":      <float_or_null>,
    "bwd_peak":     <float_or_null>,
    "step_geomean": <float_or_null>
  }},
  "representative_shapes": [ ... ],
  "git_commit": "<hash_or_null>",
  "notes": "<short>"
}}
```

Populate both Forward and Backward TFLOPS when the kernel has a backward
path. For inference-only kernels, set the Backward fields to `null` — the
trend file will render them as `-`. `trend_row.fwd_peak` / `bwd_peak` are
the per-shape maximums across PASS rows; `fwd_avg` / `bwd_avg` match the
aggregate geomeans. `step_geomean` is `sqrt(fwd_avg * bwd_avg)` (or equal
to `fwd_avg` when `bwd_avg` is null); you may leave it `null` and the
orchestrator will compute it.

Only Write / Edit / Bash (for the test + benchmark commands) / Read tools
are permitted. No chat. Do NOT modify the kernel source in this phase.
