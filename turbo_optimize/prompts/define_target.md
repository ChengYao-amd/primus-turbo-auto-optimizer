You are driving the DEFINE_TARGET phase of the kernel-optimize skill loop.

Authoritative skill excerpt (read carefully, do not alter its semantics):

<skill>
{skill_excerpt}
</skill>

User instruction for this new campaign:

<user_instruction>
{user_prompt}
</user_instruction>

Project skill directory (contains build / test / benchmark metadata):
  {project_skill_path}

Campaign directory prepared by the orchestrator (already exists):
  {campaign_dir}

State output path (where you MUST write the structured result):
  {phase_result_path}

Manifest draft path (where you MUST write manifest.yaml):
  {manifest_path}

CLI overrides (AUTHORITATIVE — copy these values verbatim into both
`manifest.yaml` and the JSON result below; do NOT substitute skill defaults,
null, or inferred values):

  max_iterations: {cli_max_iterations}
  max_duration:   {cli_max_duration}
  base_branch:    {cli_base_branch}

If any override is `null`, write YAML `null` (not `~`, not the string
"null"). Never change these three fields on your own initiative.

Forced git policy (already enforced by the orchestrator, DO NOT write
these keys into manifest.yaml):

  git_commit: true   — every ACCEPTED round is committed so rollback
                       can use `git reset --hard`. File-copy rollback
                       alone cannot restore nested subdirs, delete
                       newly-added files, or clear Triton/pycache
                       artefacts.
  git_branch: auto   — the PREPARE_ENVIRONMENT phase creates an
                       `optimize/<campaign_id>` branch off `base_branch`
                       so experiments never land on the user's source
                       branch.

Your tasks:

1. Read `{project_skill_path}/SKILL.md` and pull out every field listed under
   the kernel-optimize "Prerequisite Information" table. If anything is
   missing, keep walking the project skill's referenced files until you have
   it. Do not invent values.
2. Map the user instruction onto the `target_*` / `execution_mode` /
   `primary_metric` parameters per the DEFINE_TARGET table in the
   skill excerpt above. Do NOT emit `git_commit` or `git_branch` keys
   in the manifest — the orchestrator applies them as fixed policy.
3. Write a complete `manifest.yaml` at `{manifest_path}` using the template
   from the skill excerpt (lines near "Write manifest.yaml"). Required
   fields: target_op, target_backend, target_lang, target_gpu,
   execution_mode, project_skill, performance_target, primary_metric,
   target_shapes, kernel_source, test_command, benchmark_command,
   quick_command, profile_command, representative_shapes (use an empty
   list or placeholder if BASELINE has not yet selected them),
   related_work_file, base_branch, max_iterations, max_duration, created.
   - `quick_command`: write `python ${{CAMPAIGN_DIR}}/quick_test_bench.py`.
     The literal `${{CAMPAIGN_DIR}}` is a variable that the orchestrator
     expands at runtime; do NOT replace it with the actual path.
   - `profile_command`:
     write `python ${{CAMPAIGN_DIR}}/profile_op_shape.py`. Same rule —
     keep the `${{CAMPAIGN_DIR}}` variable verbatim. Even if the current
     environment lacks `rocprof`, still emit this template; the PROFILE
     phase will degrade to a warning rather than crash.
   - `related_work_file`: write `${{CAMPAIGN_DIR}}/related_work.md`.
   - `representative_shapes`: leave as an empty list `[]` (BASELINE will
     fill it).
   - `base_branch`: use the value from the CLI overrides block above.
     When the CLI value is `null`, default to `main`. This field is
     mandatory — the orchestrator's PREPARE_ENVIRONMENT gate rejects
     the campaign when it is missing.
   - `created`: today's timestamp in `YYYY-MM-DD HH:MM` format.
4. Finally, use the Write tool to emit a JSON document at
   `{phase_result_path}` with exactly this schema:

```json
{{
  "target_op": "<string>",
  "target_backend": "<string>",
  "target_lang": "<string>",
  "target_gpu": "<string>",
  "execution_mode": "repo",
  "project_skill": "{project_skill}",
  "primary_metric": "<string>",
  "performance_target": null_or_string,
  "target_shapes": "<string_or_all>",
  "kernel_source": "<string>",
  "test_command": "<string>",
  "benchmark_command": "<string>",
  "quick_command": "python ${{CAMPAIGN_DIR}}/quick_test_bench.py",
  "profile_command": "python ${{CAMPAIGN_DIR}}/profile_op_shape.py",
  "base_branch": "<from CLI override or 'main'>",
  "max_iterations": null_or_integer,
  "max_duration": null_or_string,
  "prerequisite_checklist": {{
    "kernel_source": true,
    "test_command": true,
    "benchmark_command": true,
    "quick_command_template": true,
    "profile_command_template": true,
    "base_branch": true,
    "benchmark_output_format": true,
    "scoring_rules": true,
    "execution_mode": true,
    "rebuild": true
  }},
  "notes": "<short text summarising any assumptions or open questions>"
}}
```

Guardrails:

- v1 only supports `execution_mode=repo`. If the project skill recommends
  `workspace`, still proceed with `repo` and note the deviation in `notes`.
- `max_iterations`, if set, MUST be strictly less than 120.
- Do NOT confirm with the user yourself. The orchestrator handles the
  manifest confirmation step after this phase returns.

Output nothing other than the files you Write. Do not chat.
