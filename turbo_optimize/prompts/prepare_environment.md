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
4. Install Primus-Turbo from the current workspace so BASELINE and every
   subsequent round exercises this repo's source tree as the optimization
   baseline. The working directory is the Primus-Turbo repo root, so
   install commands target it directly. Redirect all output to
   `{campaign_dir}/rounds/round-1/artifacts/primus_turbo_install.log`
   (do NOT stream the full `-v` build trace back into the chat; the log
   file is the source of truth). The `-v` editable build can take
   20-40 minutes on a clean tree, so raise the bash tool timeout
   accordingly (e.g. 3600000 ms).

   a. Detect an already-usable install. Run
      `python3 -c "import primus_turbo, os; print(os.path.dirname(primus_turbo.__file__))"`.
      If it exits zero AND the printed path starts with the absolute
      path of the repo root (resolve it once with `readlink -f .` or
      `pwd`), treat the install as a cached no-op: skip sub-steps (b)
      and (c), set `primus_turbo_installed=true` and
      `primus_turbo_install_log=null` in the JSON output, and record the
      import path in `notes`.
   b. Otherwise install dependencies and the package in editable mode.
      Both commands append to the same install log:
      - `pip3 install -r requirements.txt > {campaign_dir}/rounds/round-1/artifacts/primus_turbo_install.log 2>&1`
      - `env GPU_ARCHS="<target_gpu>" pip3 install --no-build-isolation -e . -v >> {campaign_dir}/rounds/round-1/artifacts/primus_turbo_install.log 2>&1`
        where `<target_gpu>` is `manifest.target_gpu` (e.g. `gfx942`).
        If `target_gpu` is empty, drop the `env GPU_ARCHS=...` prefix;
        the build will then cover every supported architecture and run
        significantly slower.
   c. Verify the install with
      `python3 -c "import primus_turbo, os; print(os.path.dirname(primus_turbo.__file__))"`
      and confirm the printed path resolves under the repo root. Editable
      mode (`-e`) keeps the runtime bound to this workspace tree, so
      OPTIMIZE edits to Python files take effect without reinstall;
      C++/HIP changes trigger an automatic rebuild on first import.

   If any pip invocation exits non-zero or the verification import fails,
   stop the phase, emit `primus_turbo_installed=false` in the JSON output
   together with the failing command and install log path, and SKIP the
   quick_test_bench / kernel snapshot steps below.
5. Generate a complete `{campaign_dir}/quick_test_bench.py` using the
   template from the project skill's "Quick validation" section. Leave
   `SHAPES` as an empty placeholder list (BASELINE will fill it).
6. Copy the starting kernel source into
   `{campaign_dir}/rounds/round-1/kernel_snapshot/` — this is the rollback
   root required by iteration_rules Rule 5. The destination path MUST
   end with `/` so `cp` treats it as a directory; a missing slash creates
   a stray file named `kernel_snapshot` in the parent directory.
7. Write the structured phase result to `{phase_result_path}` with this
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
  "primus_turbo_installed": true_or_false,
  "primus_turbo_install_log": "rounds/round-1/artifacts/primus_turbo_install.log_or_null",
  "primus_turbo_import_path": "<absolute_path_or_null>",
  "kernel_source_snapshotted": true_or_false,
  "notes": "<short>"
}}
```

Set `primus_turbo_install_log=null` when step 4 took the cached no-op
path; otherwise point it at the log file you just wrote. Keep
`primus_turbo_import_path` populated in both the cached and freshly
installed cases so later phases can cross-check which tree is live.

You MUST NOT modify the kernel source file itself in this phase. Only
snapshot it and scaffold the campaign directory. No chat output.
