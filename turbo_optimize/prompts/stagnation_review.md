You are running the STAGNATION_REVIEW phase.

Authoritative skill excerpt:

<skill>
{skill_excerpt}
</skill>

Trigger: `rollback_streak={rollback_streak}`, recent history shows
repeated failures.

Injected history:

<history>
{history_json}
</history>

Verified ineffective directions (avoid proposing these):

<verified_ineffective>
{verified_ineffective_json}
</verified_ineffective>

Tasks:

1. Re-examine the recent accepted rounds, failed attempts, and any
   profiler data under `{campaign_dir}/profiles/`.
2. Generate AT LEAST 3 fundamentally different new directions, covering
   at minimum two of the following categories: tile/launch parameters,
   memory layout/data movement, software pipelining/overlap, occupancy/
   register/LDS, backend/reference comparison, algorithm-level change.
3. **Write** the structured phase result to `{phase_result_path}` using
   the `Write` tool — not as chat / text output. The orchestrator only
   reads the file; any JSON that is only echoed in the assistant
   message is discarded and the phase is treated as failed.

```json
{{
  "new_directions": [
    {{
      "title": "<string>",
      "category": "<tile|memory|pipeline|resource|backend|algorithm>",
      "hypothesis": "<string>",
      "expected_signal": "<string>",
      "risk": "<string>"
    }}
  ],
  "primary_pick": "<index into new_directions>",
  "notes": "<short>"
}}
```

No code changes. No chat.
