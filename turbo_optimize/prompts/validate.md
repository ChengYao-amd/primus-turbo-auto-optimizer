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

{workspace_hygiene_block}
Tasks, in order:

1. Correctness gate:
   - If `{validation_level}=quick`, run `quick_command` against the
     representative shapes.
   - If `{validation_level}=full`, run `test_command` followed by
     `benchmark_command` across all target shapes.
   All shapes must pass; if any fails, record `correctness_ok=false` and
   stop (Python will trigger rollback).
2. Collect the benchmark CSV (if produced) at
   `{campaign_dir}/rounds/round-{round_n}/artifacts/benchmark.csv`. Use
   `mv` to relocate any CSV the benchmark command dumped in the working
   directory (do NOT `cp`, see <workspace_hygiene>). Save raw
   stdout/stderr to `benchmark.log` in the same folder.
3. Use `mcp__turbo__parse_bench_csv` to produce a structured score vector
   and aggregate score. Do NOT hand-aggregate; rely on the MCP tool.
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
