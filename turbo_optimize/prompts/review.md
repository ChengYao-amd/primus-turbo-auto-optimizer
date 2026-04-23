You are running the REVIEW phase for round-{round_n} in
**review_tolerant** mode.

This phase runs **after** VALIDATE (full) concluded the round is
numerically ACCEPT-flavoured. Your job is to check whether the
measured gain is the one the hypothesis actually predicted, and to
downgrade the decision when the two do not line up.

Tolerant mode is deliberately lenient: only the three *hard* rules
(hypothesis_metric_alignment, off_target_gain, correctness_bit_identity)
can force a downgrade. The two *soft* rules (quick_vs_full_agreement,
noise_band) are surfaced for context but do **not** on their own turn
an ACCEPT into a ROLLBACK.

Inputs (do NOT re-run any benchmark; trust the structured data below):

<hypothesis>
{hypothesis_json}
</hypothesis>

<optimize_result>
{optimize_result_json}
</optimize_result>

<validate_quick_result>
{validate_quick_json}
</validate_quick_result>

<validate_full_result>
{validate_full_json}
</validate_full_result>

<decision>
{decision_json}
</decision>

Pre-computed Python signals (authoritative for the numeric fields —
the shown `tolerant_verdict` is a preliminary answer you may agree
with, refine, or override):

<python_review_signals>
{review_signals_json}
</python_review_signals>

Campaign context:
- target_op:      {target_op}
- target_backend: {target_backend}
- primary_metric: {primary_metric}
- campaign_dir:   {campaign_dir}
- round:          {round_n}

Artefact pointers (Read-only; don't modify):
- round_dir:      {campaign_dir}/rounds/round-{round_n}/
- quick_csv:      {quick_csv_path}
- full_csv:       {full_csv_path}

## The five rules

Use the following definitions to produce a per-rule verdict. For each
rule you MUST return one of `"pass"`, `"warn"`, `"block"` and a short
note that cites either a value from the structured inputs above or a
line from one of the CSV artefacts.

1. **hypothesis_metric_alignment** (hard):
   Parse the hypothesis text for a target metric (`Forward TFLOPS`,
   `Backward TFLOPS`, `step_geomean`) and a predicted gain magnitude
   (`+3%`, `+5%`, ...). Compare against the observed
   `improvement_pct` / σ reported by the Python signal. `block` when
   the predicted axis gain is less than 25% of the predicted magnitude
   OR is below 2σ of the measurement stddev. If the hypothesis does
   not name a quantified target, return `pass`.

2. **off_target_gain** (hard):
   From `optimize_result.modified_files`, attribute the gain axis. A
   filename mentioning `fwd` / `forward` → Forward; `bwd` / `backward`
   / `variable_k` → Backward. Kernels whose name clearly touches a
   single axis AND whose improvement is dominated by the *other* axis
   → `block`. Files that touch both axes, or whose name encodes
   neither, → `pass` with a note explaining why attribution is
   ambiguous.

3. **quick_vs_full_agreement** (soft):
   Compare the signs of `quick_delta_pct` vs `full_delta_pct` for
   every primary metric. Any sign flip → `warn`. Equal signs (both
   non-zero) → `pass`. If `full_candidate` is unavailable (BASELINE
   only ran quick) → `pass` with a note.

4. **noise_band** (soft):
   Check the observed gain vs the per-shape stddev. Gain < 2% absolute
   OR gain < 2σ → `warn`. Otherwise → `pass`.

5. **correctness_bit_identity** (hard):
   If the hypothesis claims "bit-identical" / "numerically equivalent"
   / "same output" / "preserves numerics", and the minimum `out_snr` /
   `da_snr` / `db_snr` observed across `validate_quick_result` +
   `validate_full_result` is below 80 dB, → `block`. Otherwise
   → `pass`. When the kernel has no SNR columns (rowwise / blockwise)
   and the hypothesis does not claim equivalence → `pass`.

## Verdict table (tolerant mode)

Map the rule outcomes to a single `review_verdict`:

| hard failures                                   | verdict                          |
| ----------------------------------------------- | -------------------------------- |
| correctness_bit_identity blocks                 | `ESCALATE_HUMAN`                 |
| alignment AND off_target both block             | `DOWNGRADE_TO_ROLLBACK`          |
| alignment blocks alone                          | `DOWNGRADE_TO_NOISE_BOUND`       |
| off_target blocks alone                         | `DOWNGRADE_TO_NOISE_BOUND`       |
| no hard failures (soft warns are fine)          | `AGREE`                          |

Soft warns (quick_vs_full_agreement / noise_band) NEVER force a
downgrade in tolerant mode; they are recorded for operators to read
in the round summary.

## Output contract

Write ONE JSON document at `{phase_result_path}` with this exact
schema. No chat.

```json
{{
  "round": {round_n},
  "review_mode": "tolerant",
  "review_verdict": "AGREE | DOWNGRADE_TO_NOISE_BOUND | DOWNGRADE_TO_ROLLBACK | ESCALATE_HUMAN",
  "review_reason": "<one sentence explaining the verdict, citing specific rule names>",
  "rule_verdicts": {{
    "hypothesis_metric_alignment": {{"verdict": "pass|warn|block", "note": "<string>"}},
    "off_target_gain":             {{"verdict": "pass|warn|block", "note": "<string>"}},
    "quick_vs_full_agreement":     {{"verdict": "pass|warn|block", "note": "<string>"}},
    "noise_band":                  {{"verdict": "pass|warn|block", "note": "<string>"}},
    "correctness_bit_identity":    {{"verdict": "pass|warn|block", "note": "<string>"}}
  }},
  "python_signals": "<verbatim copy of the python_review_signals input above>",
  "notes": "<short free-form summary (<= 4 lines) for the human reader>"
}}
```

Rules:
- `review_verdict` MUST be one of the four literals above.
- When you agree with the Python `tolerant_verdict`, still re-derive
  the rule verdicts from the inputs yourself — do not blindly copy
  the Python output. The point of the phase is that you look at the
  same inputs independently.
- Cite values, not impressions. "Backward +0.08% vs predicted +3%" is
  acceptable; "the gain looks off" is not.
- Stay inside the read-only artefacts listed above. Do NOT run any
  benchmark / test / build command in this phase.
