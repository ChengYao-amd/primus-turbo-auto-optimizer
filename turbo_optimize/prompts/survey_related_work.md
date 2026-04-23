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

Scope — ONLY three source classes are in-bounds:

1. **Local kernel source tree.** The `manifest.kernel_source` path and
   everything it imports under this repo. Include sibling kernels that
   already solve adjacent shapes (e.g. other precisions, other ops on
   the same GPU) so transferable ideas are visible.
2. **AMD ROCm / Triton / CK official documentation.** Hardware guides
   (CDNA / RDNA ISA, `matrix_core_`, `rocprof` user guides), the
   compiler docs under the ROCm repo, and published tuning notes from
   AMD. Treat these as ground truth.
3. **Peer-reviewed or arxiv papers** about kernel optimization on the
   relevant architectures.

Hard exclusions — do NOT open or cite any of these:

- Other campaigns under `agent/workspace/` (including the last three
  runs). Past campaigns already contributed their lessons through
  `history.md` + `tips_for_next_run.md`; re-reading them here causes
  the doc to bloat and introduces stale context.
- Random GitHub repos outside ROCm / AMD / Triton official orgs.
- Vendor blog posts that do not carry reproducibility details
  (no shape, no GPU, no command).

Tasks:

1. Walk the three in-bounds sources in the order above. Time-box: at
   most 30 minutes of tool calls total across the three.
2. If you must clone a repository from source class (2) for inspection,
   put it under `agent/tmp/{campaign_id}/related-work/repos/`. This
   directory is ephemeral and MUST NOT be referenced from the accepted
   lineage.
3. Write `{related_work_path}` using the structure of `{template_path}`
   with these four sections and nothing else:
   - **Local kernel survey** — current implementation plus nearby
     kernels, one line each.
   - **ROCm / compiler notes** — the doc pages you read, each with one
     quote relevant to this kernel.
   - **Papers** — cited work with shape / GPU / reported metric.
   - **Shortlist of directions** — 5-10 concrete ideas this campaign
     could try, each tagged `[local]` / `[rocm]` / `[paper]`.

   Keep the whole file under 300 lines. If a section wants to grow
   past that, trim to the top findings and offload the raw notes to
   `{campaign_dir}/related_work_extra.md` (NOT accepted lineage, NOT
   referenced from the shortlist).
4. Write the structured phase result to `{phase_result_path}` with:

```json
{{
  "related_work_path": "{related_work_path}",
  "shortlist_directions": ["<direction>", ...],
  "sources": [
    {{"name": "<string>", "url_or_path": "<string>", "source_class": "local|rocm|paper", "performance_claim": "<string_or_null>"}}
  ],
  "excluded": ["<source you considered but rejected as out-of-scope>", ...],
  "caveats": ["<string>", ...],
  "notes": "<short>"
}}
```

No chat. The only outputs are the two files above.
