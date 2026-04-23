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
## Measurement-consistency contract

Every per-round VALIDATE reads one CSV from
`rounds/round-N/artifacts/benchmark.csv` and compares it shape-by-shape
against round-1's version at
`rounds/round-1/artifacts/benchmark.csv`. For that comparison to be
methodologically sound the two CSVs MUST:

* contain the **same set of representative shapes** (identical `B/M/N/K`),
* use the **same benchmark harness** (`manifest.quick_command`), and
* follow the **same CSV schema** (so `mcp__turbo__parse_bench_csv`
  parses them identically).

BASELINE therefore does NOT use `benchmark_command` output as the
authoritative score source. That full sweep still runs — it is the
coverage gate and the shape picker — but its CSV lives under
`full_benchmark.csv` and is only referenced from the structured output
via `reference_benchmark_csv`. The per-round-comparable CSV is always
the quick bench.

Tasks (strict order):

1. Run the focused test command from `manifest.test_command`. All tests
   MUST pass. If any fails, stop, record the failure reason in the JSON
   output (`test_pass=false`) and do NOT proceed to benchmarking.
2. Run the focused benchmark command from `manifest.benchmark_command`
   as the **coverage sweep** used only to pick representative shapes.
   Capture stdout / stderr to
   `{campaign_dir}/rounds/round-1/artifacts/full_benchmark.log`.
   If the project emits a CSV next to the current working directory,
   `mv` it to
   `{campaign_dir}/rounds/round-1/artifacts/full_benchmark.csv` (do NOT
   `cp` — see <workspace_hygiene>). This CSV will NOT drive the
   aggregate score or the trend row; it is archival + reference only.
3. Select 3-5 representative shapes covering small / medium / large
   behaviour from the PASS rows of `full_benchmark.csv`. These will
   drive `quick_command` and every subsequent per-round measurement.
4. Update the two files that must stay in sync with the chosen
   representative shapes:
   - `{campaign_dir}/quick_test_bench.py`: replace the empty `SHAPES`
     placeholder with the list you selected.
   - `{campaign_dir}/manifest.yaml`: set `representative_shapes` to the
     same list. Do not modify any other field unless explicitly required.
5. Run `manifest.quick_command` once against the freshly filled
   `representative_shapes` and MAKE IT EMIT THE AUTHORITATIVE CSV.
   The quick bench script MUST be invoked with a `--summary-csv`
   argument pointing at
   `{campaign_dir}/rounds/round-1/artifacts/benchmark.csv`. The
   script's combined stdout+stderr go to
   `{campaign_dir}/rounds/round-1/artifacts/quick_baseline.log` (use
   `2>&1 | tee` or shell redirection; do NOT truncate after the
   fact). Example:

       python {campaign_dir}/quick_test_bench.py \
         --summary-csv {campaign_dir}/rounds/round-1/artifacts/benchmark.csv \
         --csv {campaign_dir}/rounds/round-1/artifacts/benchmark_per_repeat.csv \
         2>&1 | tee {campaign_dir}/rounds/round-1/artifacts/quick_baseline.log

   The `--summary-csv` output is the single source of truth for the
   BASELINE aggregate and for the per-shape regression gate in every
   future VALIDATE. All shapes in this run MUST PASS; if any
   correctness check fails, stop, emit `test_pass=false`, and leave
   `quick_baseline_log` / `benchmark_csv` set to the partial paths so
   the failure is debuggable.
6. Parse `rounds/round-1/artifacts/benchmark.csv` via
   `mcp__turbo__parse_bench_csv` (DO NOT hand-aggregate). Use the
   returned `aggregate` and `rows` to populate `aggregate_score`,
   `score_vector`, and `trend_row` in the JSON output below. This is
   the **same tool** VALIDATE calls each round, which is how we
   guarantee identical geomean formulas, identical shape keys, and
   identical stddev interpretation across round-1 and round-N.
7. Write `{campaign_dir}/rounds/round-1/summary.md` using the canonical
   round-summary template from the skill excerpt. Round-1's `Single
   change` block must explicitly say "No code change. Baseline round."
   and its `Decision` section must be `BASELINE`.
8. Copy the current kernel source into
   `{campaign_dir}/rounds/round-1/kernel_snapshot/` — note the trailing
   `/` so `cp` treats the destination as a directory (PREPARE_ENVIRONMENT
   may have done this already; verify and repair if missing).
9. Do NOT touch `{campaign_dir}/logs/optimize.md` or
   `{campaign_dir}/logs/performance_trend.md` in this phase. The
   orchestrator owns both files end-to-end: the BASELINE block in
   `optimize.md` and the BASELINE row in `performance_trend.md` are
   both written by Python from the JSON below, so any manual edit
   here would collide and produce duplicate / inconsistent rows.
10. Emit the structured phase result at `{phase_result_path}`:

```json
{{
  "test_pass": true_or_false,
  "benchmark_csv": "rounds/round-1/artifacts/benchmark.csv",
  "reference_benchmark_csv": "rounds/round-1/artifacts/full_benchmark.csv",
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

`benchmark_csv` MUST point at the quick-harness CSV produced in step
5; `reference_benchmark_csv` points at the full-sweep CSV from step 2.
`aggregate_score` / `score_vector` / `trend_row` are always computed
from `benchmark_csv`, never from the full sweep.

Populate both Forward and Backward TFLOPS when the kernel has a backward
path. For inference-only kernels, set the Backward fields to `null` — the
trend file will render them as `-`. `trend_row.fwd_peak` / `bwd_peak` are
the per-shape maximums across PASS rows; `fwd_avg` / `bwd_avg` match the
aggregate geomeans. `step_geomean` is `sqrt(fwd_avg * bwd_avg)` (or equal
to `fwd_avg` when `bwd_avg` is null); you may leave it `null` and the
orchestrator will compute it.

Only Write / Edit / Bash (for the test + benchmark commands) / Read tools
are permitted. No chat. Do NOT modify the kernel source in this phase.
