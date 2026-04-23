You are running the PROFILE phase for the current kernel.

Authoritative skill excerpt:

<skill>
{skill_excerpt}
</skill>

Campaign context:
- campaign_dir:    {campaign_dir}
- round_n:         {round_n}
- trigger:         {trigger}
- target_op:       {target_op}
- target_backend:  {target_backend}
- target_gpu:      {target_gpu}
- profile_command: {profile_command}
- representative_shape_hint: {representative_shape_hint}

Trigger meaning:
- `post_baseline` — we just established the baseline; capture a
  profile that future rounds can diff against.
- `post_accept`   — the previous round was accepted; record the new
  baseline profile at `profiles/round-{round_n}/`.
- `pre_stagnation` — two rollbacks in a row just happened; the agent
  needs a fresh profile so STAGNATION_REVIEW has current counters.

{workspace_hygiene_block}
Tasks, in order:

1. Pick ONE shape from `manifest.representative_shapes` (fall back to
   the first entry of `manifest.target_shapes` if the representative
   list is empty). Record which shape you picked. If the campaign is
   GPU-less (shouldn't happen in this repo) or `profile_command` is
   empty, emit the JSON with `skipped=true` and stop.

2. Check tool availability:
   - `command -v rocprofv3` → record `rocprofv3_available`
   - `command -v rocprof-compute` → record `rocprof_compute_available`
   If NEITHER is available, emit `skipped=true` with
   `skip_reason="rocprof tools missing"` and stop. Do NOT try to `pip
   install` them.

3. For each available tool, run the profile on the chosen shape and
   write artifacts under
   `{campaign_dir}/profiles/round-{round_n}_{trigger}/`:
   - `rocprofv3 --kernel-trace --output-format csv json -d <out>
     -- {profile_command} --shape '<json>' --out-dir <out>` →
     produces `results.csv` + `timeline.json`.
   - `rocprof-compute profile --name round-{round_n} --out <out> --
     {profile_command} --shape '<json>' --out-dir <out>` →
     produces the rocprof-compute workload directory.
   Cap each invocation with a 5-minute timeout. If the tool exits
   non-zero, capture stderr to `<tool>.err` and mark that tool's
   status as `failed`.

4. Write a concise `profile_summary.md` next to the raw artifacts
   covering:
   - the shape that was profiled
   - the top-5 kernels by duration
   - occupancy + LDS usage for the target kernel
   - 1-3 observations relevant to `target_op` (e.g. "load_B_shared
     stall = 62%").
   Keep the whole file under 80 lines.

5. Emit the structured phase result at `{phase_result_path}`:

```json
{{
  "round": {round_n},
  "trigger": "{trigger}",
  "skipped": true_or_false,
  "skip_reason": "<string_or_null>",
  "shape_profiled": {{...}},
  "artifacts_dir": "profiles/round-{round_n}_{trigger}",
  "summary_path": "profiles/round-{round_n}_{trigger}/profile_summary.md",
  "tools": [
    {{
      "name": "rocprofv3",
      "available": true_or_false,
      "status": "ok|failed|skipped",
      "command": "<string>",
      "artifacts": ["<relative_path>", ...]
    }},
    {{
      "name": "rocprof-compute",
      "available": true_or_false,
      "status": "ok|failed|skipped",
      "command": "<string>",
      "artifacts": ["<relative_path>", ...]
    }}
  ],
  "top_kernels": [
    {{"name": "<string>", "duration_us": <float>, "invocations": <int>}}
  ],
  "observations": ["<string>", ...],
  "notes": "<short>"
}}
```

Forbidden:
- Do NOT edit the kernel source in this phase.
- Do NOT run the full benchmark; the profile covers a single shape.
- Do NOT mutate `logs/optimize.md` or the performance trend; that is
  the orchestrator's job.

No chat output.
