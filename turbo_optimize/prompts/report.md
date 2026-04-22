You are running the REPORT phase. The campaign is terminating.

Authoritative skill excerpt:

<skill>
{skill_excerpt}
</skill>

Injected history snapshot:

<history>
{history_json}
</history>

Termination decision (produced by the orchestrator):

<termination>
{termination_json}
</termination>

Campaign context for tip scoping:
- target_op:      {target_op}
- target_backend: {target_backend}
- target_gpu:     {target_gpu}
- tips_path:      {tips_path}

Tasks (in order):

1. Append a `## Final Report` section to `{campaign_dir}/logs/optimize.md`
   containing:
   - Baseline aggregate score(s)
   - Final best aggregate score(s) + percentage improvement vs baseline
   - Total rounds, accepted rounds, rollback rounds
   - Key effective optimizations with round numbers
   - Verified ineffective directions summary
   - Top-three recommended next steps (if further optimization is worthwhile)

2. Distil historical tips and append them to
   `agent/historical_experience/<target_gpu>/<target_op>/<target_backend>/tips.md`
   via `mcp__turbo__append_tip` (do NOT use `Write` directly — the MCP
   tool holds the write lock for concurrent campaigns). **Purpose:** the
   tips file is read by the NEXT campaign (possibly on a different GPU
   or operator) so future agents skip failure paths we already verified
   and re-apply patterns that consistently worked. Quality matters more
   than quantity.

   Inclusion criteria — append at most 5 tips total, split between the
   two categories. Skip the category entirely if nothing qualifies:

   A. **Failure tips** (prevent repeated test cost):
      - The direction was tried in THIS campaign, rolled back, and we
        have a concrete profiler / benchmark signal that explains why.
      - Signal must be something a future agent can re-check without
        re-running the full round (e.g. "Triton does not fuse xxx when
        `BLOCK_K > 64` on gfx942 → spills to LDS → FWD drops ~20%").
      - Skip if the failure was a local bug (compile error, off-by-one,
        shape typo). Those are not reusable lessons.

   B. **Success tips** (cross-op reusable patterns):
      - An accepted direction that is NOT tied to this specific shape /
        kernel path, i.e. plausibly useful for another op on the same
        (backend, gpu) combination.
      - Prefer patterns tied to hardware constraints, compiler quirks,
        or algorithmic primitives (e.g. "on gfx950 MFMA 16x16x32 beats
        32x32x16 once K ≥ 128, regardless of M/N") over kernel-specific
        magic numbers.
      - Skip if the improvement only holds on one shape family from
        this campaign's `target_shapes` — that belongs in summary.md,
        not in the cross-campaign knowledge base.

   Quality bar for EVERY tip (reject if any is missing):
   - `context`: hardware scope + op class + backend version. NOT this
     campaign's campaign_id or round number.
   - `signal`: the observable that proves the claim. Must cite a metric
     name, profiler counter, or a specific error pattern.
   - `takeaway`: the reusable lesson in one sentence, phrased as a
     constraint or rule of thumb, not as a narrative.
   - `applicability`: explicit WHEN-to-reuse + WHEN-NOT-to-reuse. If you
     cannot name a case where the tip does not apply, you have not
     distilled it enough.

   Forbidden in tip bodies:
   - Shape-specific magic numbers without the constraint that produced
     them (e.g. "use BLOCK_M=128" without "on K-major layout with K ≥
     256"). Rule of thumb: a reader optimizing a different op on the
     same hardware should still benefit.
   - Phrases like "we tried X" / "after round-3" — the tip must read as
     a standing lesson, not a campaign diary entry.
   - Claims that do not appear anywhere in `{campaign_dir}/logs/`. Every
     tip must be traceable to an artifact in this campaign.

   Call pattern per tip:

   ```
   mcp__turbo__append_tip(
     op="{target_op}",
     backend="{target_backend}",
     gpu="{target_gpu}",
     round=<round_n that produced the evidence>,
     status="ACCEPTED" | "ROLLED_BACK",
     context="<hardware + op + backend scope>",
     signal="<observable>",
     takeaway="<one-line standing lesson>",
     applicability="<when to reuse + when not to>"
   )
   ```

3. Emit structured phase result at `{phase_result_path}`:

```json
{{
  "baseline_aggregate": {{ ... }},
  "final_best_aggregate": {{ ... }},
  "improvement_pct": {{ ... }},
  "total_rounds": <int>,
  "accepted_rounds": <int>,
  "rollback_rounds": <int>,
  "key_effective": ["round-N: <description>", ...],
  "verified_ineffective": ["<string>", ...],
  "recommended_next": ["<string>", ...],
  "termination_condition": "<T1|T2|T3|T4|T5>",
  "tips_appended": [
    {{
      "category": "failure | success",
      "round": <int>,
      "status": "ACCEPTED | ROLLED_BACK",
      "takeaway": "<one-line>",
      "applicability": "<when / when not>"
    }}
  ],
  "notes": "<short>"
}}
```

`tips_appended` records only the tips you actually wrote via
`append_tip`. Empty list is acceptable and is the correct outcome when
the campaign produced no cross-op-reusable lessons.

No further code changes. No chat output.
