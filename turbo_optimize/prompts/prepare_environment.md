You are running the PREPARE_ENVIRONMENT phase.

Authoritative skill excerpt:

<skill>
{skill_excerpt}
</skill>

Confirmed manifest:

<manifest>
{manifest_yaml}
</manifest>

Campaign directory: `{campaign_dir}`

{workspace_hygiene_block}
Tasks, in order:

1. If `git_branch != "none"`, run the appropriate `git checkout -b`. Use
   the exact branch name from manifest.git_branch (`auto` → create
   `optimize/<campaign_id>`, where campaign_id is `{campaign_id}`).
2. Ensure the full campaign directory tree exists, per the skill excerpt:
   `logs/`, `profiles/`, `rounds/round-1/{{kernel_snapshot,artifacts}}`.
3. Initialise (or leave intact) the baseline log skeleton. The orchestrator
   has already created `logs/optimize.md` and `logs/performance_trend.md`
   with valid headers; you MUST NOT overwrite them.
4. Generate a complete `{campaign_dir}/quick_test_bench.py` using the
   template from the project skill's "Quick validation" section. Leave
   `SHAPES` as an empty placeholder list (BASELINE will fill it).
5. Copy the starting kernel source into
   `{campaign_dir}/rounds/round-1/kernel_snapshot/` — this is the rollback
   root required by iteration_rules Rule 5. The destination path MUST
   end with `/` so `cp` treats it as a directory; a missing slash creates
   a stray file named `kernel_snapshot` in the parent directory.
6. Write the structured phase result to `{phase_result_path}` with this
   schema:

```json
{{
  "git_branch_created": "<branch_or_null>",
  "campaign_dir": "{campaign_dir}",
  "generated": [
    "logs/optimize.md",
    "logs/performance_trend.md",
    "rounds/round-1/kernel_snapshot/<files>",
    "quick_test_bench.py"
  ],
  "kernel_source_snapshotted": true_or_false,
  "notes": "<short>"
}}
```

You MUST NOT modify the kernel source file itself in this phase. Only
snapshot it and scaffold the campaign directory. No chat output.
