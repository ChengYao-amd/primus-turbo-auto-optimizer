You are running the READ_HISTORICAL_TIPS phase.

Authoritative skill excerpt:

<skill>
{skill_excerpt}
</skill>

Manifest (read-only):

<manifest>
{manifest_yaml}
</manifest>

Tips path convention:
  agent/historical_experience/<target_gpu>/<target_op>/<target_backend_lower>/tips.md

For this campaign it resolves to:
  {tips_path}

You have access to MCP tools under the `mcp__turbo__*` namespace. In
particular, `mcp__turbo__query_tips` returns structured entries parsed
from the tips file and is preferred over raw markdown reading for this
phase.

Tasks:

1. Call `mcp__turbo__query_tips` with the campaign's op / backend / gpu
   and summarise any applicable lessons. If the file does not exist, note
   that and continue — it is not an error.
2. Write the structured phase result to `{phase_result_path}`:

```json
{{
  "tips_path": "{tips_path}",
  "file_present": true_or_false,
  "applicable_entries": [
    {{"heading": "<string>", "summary": "<string>"}}
  ],
  "skip_reason": "<string_or_empty>"
}}
```

No chat output. Do not modify the tips file in this phase; new entries
are distilled and appended by the REPORT phase at campaign end, where
Claude can compare all accepted and rolled-back rounds before choosing
which lessons are cross-op reusable.
