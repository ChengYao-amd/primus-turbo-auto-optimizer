You are running the VALIDATE phase for round-{round_n}.

Authoritative skill excerpt:

<skill>
{skill_excerpt}
</skill>

Scoring reference:

<scoring>
{scoring_excerpt}
</scoring>

Iteration rules (Rule 3, Rule 6, Rule 8):

<rules>
{rules_excerpt}
</rules>

Campaign context:
- test_command:      {test_command}
- benchmark_command: {benchmark_command}
- quick_command:     {quick_command}
- primary_metric:    {primary_metric}
- round_dir:         {campaign_dir}/rounds/round-{round_n}/

Current best from orchestrator-injected history:

<current_best>
{current_best_json}
</current_best>

Validation level for this round: `{validation_level}` (`quick` by default;
the orchestrator upgrades to `full` when the direction completes, the
improvement is near noise, or the change is high-risk).

## Measurement-consistency contract

`{campaign_dir}/rounds/round-{round_n}/artifacts/benchmark.csv` is the
single source of truth for this round's `aggregate_score`,
`score_vector`, and `trend_row`. It MUST be produced by the SAME
`quick_command` harness that BASELINE used, over the SAME
`representative_shapes`, regardless of whether `{validation_level}` is
`quick` or `full`. The orchestrator's per-round regression gate
compares this CSV shape-by-shape against round-1's
`benchmark.csv`; any schema or shape-set drift silently disables the
gate.

When `{validation_level}=full` you MAY also run `benchmark_command` as
an archival coverage sweep and save the result to
`full_benchmark.csv` / `full_benchmark.log` — but that CSV must not be
used for the score. See step 1 below.

{workspace_hygiene_block}
Tasks, in order:

1. Correctness + measurement gate:
   - If `{validation_level}=quick`, run `quick_command` against the
     representative shapes. Invoke the quick bench script with
     `--summary-csv {campaign_dir}/rounds/round-{round_n}/artifacts/benchmark.csv`
     so the canonical CSV lands in the right place. Save combined
     stdout+stderr to `benchmark.log` in the same folder.
   - If `{validation_level}=full`, first run `test_command` to confirm
     correctness across the full test suite, then run
     `benchmark_command` across all target shapes (save its output to
     `full_benchmark.csv` / `full_benchmark.log`). After the full
     sweep, ALSO run `quick_command` once with the same
     `--summary-csv` pointing at `benchmark.csv` so the authoritative
     score is produced by the same harness BASELINE used.
   All shapes in the quick bench MUST PASS; if any check fails, record
   `correctness_ok=false` and stop (Python will trigger rollback).
2. Relocate any stray CSVs the benchmark commands dumped in the working
   directory into the artifacts folder using `mv` (do NOT `cp`, see
   <workspace_hygiene>). After step 1 the folder must contain at least:
   - `benchmark.csv` — authoritative, quick-harness schema, one row per
     representative shape.
   - `benchmark.log` — combined stdout/stderr of the quick bench.
   and for `{validation_level}=full`:
   - `full_benchmark.csv` / `full_benchmark.log` — archival full sweep
     (not parsed for scoring).
3. Use `mcp__turbo__parse_bench_csv` to parse `benchmark.csv` ONLY.
   Never hand-aggregate and never substitute the full-sweep CSV: using
   a different CSV across rounds breaks the methodology contract
   above.
4. Write `{campaign_dir}/rounds/round-{round_n}/summary.md` using the
   Round Summary Template from the skill excerpt. Populate:
   - Hypothesis / Single change (from ANALYZE + OPTIMIZE outputs)
   - Results table (per shape)
   - Aggregate section
   - Decision placeholder — the orchestrator (Python) will finalise
     ACCEPT / ROLLED BACK after calling scoring.py.
5. **Classify any failure BEFORE writing the JSON.** When
   `correctness_ok` or `build_ok` is false (or about to be false), pick
   exactly one `failure_category` from:
   - `build_compile` — compiler / JIT rejected the source; captured in
     the build log.
   - `build_link` — link-time or codegen error; ROCm / HIP link log is
     the source of truth.
   - `runtime_assert` — kernel launched, then aborted via
     `HIP_CHECK` / `TORCH_CHECK` / Python-level assert.
   - `runtime_oom` — GPU memory exhaustion (look for
     `hipErrorOutOfMemory` / `CUDA out of memory`).
   - `runtime_hang` — kernel did not return within the harness timeout.
   - `snr_fail` — kernel executed but SNR / max-abs-diff exceeded
     threshold in `quick_test_bench.py`.
   - `bench_regression` — all correctness checks pass but the
     benchmark CSV still flagged `Check=FAIL` on at least one row.
   - `other` — anything else; MUST include `failure_summary` detailing
     the symptom.
   Also fill `failure_summary` (<= 3 sentences) with: what the
   observed symptom was, which log path proves it, and one candidate
   hypothesis for what to try in the retry. This field is injected
   verbatim into the next OPTIMIZE prompt — keep it mechanical.
6. Emit structured phase result at `{phase_result_path}`:

```json
{{
  "round": {round_n},
  "validation_level": "{validation_level}",
  "correctness_ok": true_or_false,
  "build_ok": true_or_false,
  "failure_category": "<one_of_the_categories_above_or_null_when_all_ok>",
  "failure_summary": "<<=3_sentences_or_null>",
  "failure_log_path": "<relative_path_or_null>",
  "benchmark_csv": "rounds/round-{round_n}/artifacts/benchmark.csv",
  "score_vector": [
    {{"shape": {{...}}, "check": "PASS", "metrics": {{
      "Forward TFLOPS": <float>,
      "Backward TFLOPS": <float_or_null>
    }}}}
  ],
  "aggregate_score": {{
    "Forward TFLOPS": <float>,
    "Backward TFLOPS": <float_or_null>
  }},
  "trend_row": {{
    "fwd_avg":      <float>,
    "fwd_peak":     <float>,
    "bwd_avg":      <float_or_null>,
    "bwd_peak":     <float_or_null>,
    "step_geomean": <float_or_null>
  }},
  "notes": "<short>"
}}
```

On a healthy run (`correctness_ok=true` AND `build_ok=true`), set
`failure_category` and `failure_summary` to `null`. On any failure,
both MUST be populated — the orchestrator forwards them into the
retry prompt.

Populate both Forward and Backward TFLOPS when the kernel has a backward
path. For inference-only kernels, set the backward fields to `null` — the
trend file will render them as `-`. `trend_row.fwd_peak` / `bwd_peak` are
the per-shape maximums across PASS rows; `fwd_avg` / `bwd_avg` match the
aggregate geomeans. `step_geomean` is `sqrt(fwd_avg * bwd_avg)` (or equal
to `fwd_avg` when `bwd_avg` is null); you may leave it `null` and the
orchestrator will compute it.

Forbidden:
- Do NOT commit to git here. The orchestrator runs `git commit` after the
  Python-side decision confirms ACCEPT and `git_commit=true`.
- Do NOT edit `logs/optimize.md` or `logs/performance_trend.md` in this
  phase. The orchestrator appends structured rows post-decision.

No chat output.
