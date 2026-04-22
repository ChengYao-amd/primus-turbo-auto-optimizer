You are running the OPTIMIZE phase for round-{round_n}.

Authoritative skill excerpt:

<skill>
{skill_excerpt}
</skill>

Iteration rules:

<rules>
{rules_excerpt}
</rules>

Input hypothesis from ANALYZE (round-{round_n}):

<hypothesis>
{hypothesis_json}
</hypothesis>
{retry_context_block}
Campaign context:
- kernel_source:    {kernel_source}
- campaign_dir:     {campaign_dir}
- round_dir:        {campaign_dir}/rounds/round-{round_n}/
- rebuild_required: {rebuild_required}

{workspace_hygiene_block}
Tasks:

1. Make ONE meaningful code change that implements exactly the primary
   hypothesis. No mixing in cleanups, formatting, or unrelated changes.
2. If the backend requires rebuild (`rebuild_required=true`), run the
   rebuild command from the project skill. Capture output into
   `{campaign_dir}/rounds/round-{round_n}/artifacts/build.log`.
3. Copy the modified kernel file into
   `{campaign_dir}/rounds/round-{round_n}/kernel_snapshot/` so the round
   is rollback-reproducible (Rule 5). Keep the trailing `/` on the
   destination so `cp` treats it as a directory.
4. Write the structured phase result at `{phase_result_path}`:

```json
{{
  "round": {round_n},
  "modified_files": ["<path>", ...],
  "diff_summary": "<short one-liner>",
  "build_ok": true_or_false,
  "build_log": "rounds/round-{round_n}/artifacts/build.log",
  "kernel_snapshotted": true_or_false,
  "notes": "<short>"
}}
```

Forbidden:
- Do NOT run tests or benchmarks here; VALIDATE will do that.
- Do NOT commit to git; VALIDATE handles git_commit after acceptance.
- Do NOT modify files outside the kernel source path and the
  per-round artifacts folder.

No chat output.
