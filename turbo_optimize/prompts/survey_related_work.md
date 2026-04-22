You are running the SURVEY_RELATED_WORK phase.

Authoritative skill excerpt:

<skill>
{skill_excerpt}
</skill>

Manifest (read-only):

<manifest>
{manifest_yaml}
</manifest>

Campaign directory: `{campaign_dir}`
Related-work output: `{related_work_path}`
Template to follow: `{template_path}`

Tasks:

1. Survey local project implementations under the workspace root, plus AMD
   ROCm docs, SOTA open-source implementations, and competitor baselines
   for the operator `{target_op}` on backend `{target_backend}` and GPU
   `{target_gpu}`. Time-box: at most 30 minutes of tool calls.
2. If you choose to clone repositories for inspection, put them under
   `agent/tmp/{campaign_id}/related-work/repos/`. This directory is
   ephemeral and must not be referenced from the accepted lineage.
3. Write `{related_work_path}` using the structure of `{template_path}`,
   summarising:
   - reviewed implementations and libraries
   - reported performance claims + hardware / shape context
   - transferable optimization ideas worth trying in this campaign
   - reproducibility caveats
   - a concrete shortlist of optimization directions to try locally
4. Write the structured phase result to `{phase_result_path}` with:

```json
{{
  "related_work_path": "{related_work_path}",
  "shortlist_directions": ["<direction>", ...],
  "sources": [
    {{"name": "<string>", "url_or_path": "<string>", "performance_claim": "<string_or_null>"}}
  ],
  "caveats": ["<string>", ...],
  "notes": "<short>"
}}
```

No chat. The only outputs are the two files above.
