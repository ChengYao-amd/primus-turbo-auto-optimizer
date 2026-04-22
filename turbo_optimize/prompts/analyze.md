You are running the ANALYZE phase for round-{round_n}.

Authoritative skill excerpt:

<skill>
{skill_excerpt}
</skill>

Iteration rules (hard constraints):

<rules>
{rules_excerpt}
</rules>

Campaign context:
- target_op:      {target_op}
- target_backend: {target_backend}
- target_gpu:     {target_gpu}
- primary_metric: {primary_metric}
- campaign_dir:   {campaign_dir}
- round:          {round_n}

Structured history injected by the orchestrator (authoritative — do NOT
assume any different state even if `Read` shows stale cached content):

<history>
{history_json}
</history>

Forbidden hypotheses (already verified ineffective — propose different
directions):

<verified_ineffective>
{verified_ineffective_json}
</verified_ineffective>

Pending directions listed in optimize.md's "Directions to Try":

<directions_to_try>
{directions_to_try_json}
</directions_to_try>

You have access to MCP tools under `mcp__turbo__*`; prefer them over raw
markdown parsing:
  - `mcp__turbo__query_trend` for the latest trend rows
  - `mcp__turbo__list_ineffective_directions` to double-check the forbid list
  - `mcp__turbo__read_best_summary` for the current best kernel snapshot body
  - `mcp__turbo__query_tips` for reusable lessons on this op / backend / gpu

Tasks:

1. Read the current best kernel snapshot (previous accepted round), the
   related_work.md, and any profiling artifacts under `{campaign_dir}/profiles/`.
2. Classify the bottleneck per the ANALYZE table (compute / memory /
   resource / launch-overhead bound). Cite the evidence.
3. Pick ONE primary hypothesis for this round that:
   - does not overlap with any entry in `<verified_ineffective>`
   - can be implemented as a SINGLE kernel change
   - has a concrete verification signal (which metric or profiler counter
     you expect to shift)
4. Write the structured phase result at `{phase_result_path}`:

```json
{{
  "round": {round_n},
  "primary_hypothesis": "<string>",
  "bottleneck_class": "compute | memory | resource | launch",
  "expected_benefit": "<e.g. +5% geomean>",
  "risks": ["<string>", ...],
  "verification_signal": "<string>",
  "rejected_alternatives": [
    {{"direction": "<string>", "reason": "<string>"}}
  ],
  "evidence_paths": ["<relative_path>", ...]
}}
```

No chat. No code changes in this phase.
