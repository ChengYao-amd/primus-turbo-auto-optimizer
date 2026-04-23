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

1. **Base-branch + clean-tree gate** (must run before any checkout):
   a. Read `manifest.base_branch`. This is the branch every OPTIMIZE
      commit must descend from. If the field is missing or empty, stop
      the phase and emit the JSON output below with
      `base_branch_confirmed=false` plus a note that the manifest needs
      fixing — do NOT guess.
   b. Run `git rev-parse --abbrev-ref HEAD` and `git rev-parse HEAD`.
      Record both. If the current branch does not match
      `manifest.base_branch`, still record what you found and set
      `base_branch_confirmed=false`; the orchestrator will decide
      whether to continue.
   c. Run `git status --porcelain --ignore-submodules=all`. Submodules
      under `3rdparty/` are excluded on purpose: their HEAD often drifts
      from the pinned commit as part of normal development (e.g.
      `composable_kernel` tracked on a feature branch), and a plain
      `git commit` in the parent repo never folds submodule work-tree
      edits into the commit anyway. If the remaining porcelain output is
      non-empty AND `manifest.git_commit == true`, set
      `workspace_clean=false` in the JSON and stop: dirty commits on top
      of arbitrary local edits in the parent repo are unreproducible.
      `workspace_clean=true` only when the submodule-ignored porcelain
      output is empty. For audit, also run `git submodule status` and
      put the one-line summary under `submodule_state` in the JSON;
      set `submodule_state_ignored=true` to make the relaxation
      explicit.
2. If the gate above passed and `git_branch != "none"`, run the
   appropriate `git checkout -b`. Use the exact branch name from
   manifest.git_branch (`auto` → create `optimize/<campaign_id>`,
   where campaign_id is `{campaign_id}`). Branch off
   `manifest.base_branch` explicitly:
   `git checkout -b <target> <base_branch>`.
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

   Always execute both install commands — there is no cache short-circuit.
   Detecting "already installed" via `python -c "import primus_turbo"` in
   the repo root is unreliable because `sys.path[0]` equals the cwd, so
   the repo's `primus_turbo/` source directory shadows whatever is in
   `site-packages/` and the import will appear to succeed even when the
   only installed artifact is a frozen wheel. Running `pip install -e .`
   unconditionally is the only way to guarantee the runtime is pinned
   to this workspace tree.

   a. `pip3 install -r requirements.txt > {campaign_dir}/rounds/round-1/artifacts/primus_turbo_install.log 2>&1`
   b. `env GPU_ARCHS="<target_gpu>" pip3 install --no-build-isolation -e . -v >> {campaign_dir}/rounds/round-1/artifacts/primus_turbo_install.log 2>&1`
      where `<target_gpu>` is `manifest.target_gpu` (e.g. `gfx942`). If
      `target_gpu` is empty, drop the `env GPU_ARCHS=...` prefix; the
      build will then cover every supported architecture and run
      significantly slower.
   c. Verify the editable install is in effect without the cwd-shadow
      masking a stale `site-packages/` copy. Use `python3 -P` (Python
      3.11+) which disables the implicit `sys.path[0] = cwd` entry, so
      the import resolves through the normal site-packages / `.pth`
      chain even when executed from the repo root:
      `python3 -P -c "import primus_turbo, os; print(os.path.dirname(primus_turbo.__file__))"`.
      Confirm the printed path resolves under the repo root. Also run
      `pip3 show primus_turbo` and record the `Location` and `Editable
      project location` (or `Version`) fields in `notes` for audit.
      Editable mode (`-e`) keeps the runtime bound to this workspace
      tree, so OPTIMIZE edits to Python files take effect without
      reinstall; C++/HIP changes trigger an automatic rebuild on first
      import.

   If any pip invocation exits non-zero or the verification import does
   not resolve under the repo root, stop the phase, emit
   `primus_turbo_installed=false` in the JSON output together with the
   failing command and install log path, and SKIP the quick_test_bench /
   kernel snapshot steps below.
5. Generate a complete `{campaign_dir}/quick_test_bench.py` using the
   template from the project skill's "Quick validation" section. Leave
   `SHAPES` as an empty placeholder list (BASELINE will fill it).

   The generated script MUST expose a `--repeats N` CLI flag (default 3)
   and report every metric as `mean ± std` across those repeats. Use
   `torch.cuda.synchronize()` around each timed block and drop the first
   repeat as a warm-up. Emit both a machine-readable CSV with per-repeat
   rows AND a one-line summary per shape — downstream parsing depends
   on the variance column being present even when `--repeats 1`.
6. Generate `{campaign_dir}/profile_op_shape.py`, a tiny helper that
   profiles the current kernel on one representative shape via
   `rocprofv3 --kernel-trace` and `rocprof-compute profile`. The script
   MUST:
   - Accept `--shape` (JSON dict), `--out-dir`, and `--kind`
     (`rocprofv3` / `rocprof-compute`; default `both`).
   - Re-use the import path and shape helpers from
     `quick_test_bench.py` so profile + correctness run the same code.
   - Write timeline JSON + counter CSV under
     `{campaign_dir}/profiles/<round_or_tag>/`.
   This script is invoked by the PROFILE phase and manually by the
   agent during ANALYZE / STAGNATION_REVIEW.
7. Copy the starting kernel source into
   `{campaign_dir}/rounds/round-1/kernel_snapshot/` — this is the rollback
   root required by iteration_rules Rule 5. The destination path MUST
   end with `/` so `cp` treats it as a directory; a missing slash creates
   a stray file named `kernel_snapshot` in the parent directory.
8. Write the structured phase result to `{phase_result_path}` with this
   schema:

```json
{{
  "git_branch_created": "<branch_or_null>",
  "base_branch_confirmed": true_or_false,
  "base_branch_expected": "<manifest.base_branch or null>",
  "base_branch_observed": "<git rev-parse --abbrev-ref HEAD>",
  "base_commit_observed": "<git rev-parse HEAD>",
  "workspace_clean": true_or_false,
  "submodule_state_ignored": true,
  "submodule_state": "<one-line git submodule status summary>",
  "campaign_dir": "{campaign_dir}",
  "generated": [
    "logs/optimize.md",
    "logs/performance_trend.md",
    "rounds/round-1/kernel_snapshot/<files>",
    "quick_test_bench.py",
    "profile_op_shape.py"
  ],
  "primus_turbo_installed": true_or_false,
  "primus_turbo_install_log": "rounds/round-1/artifacts/primus_turbo_install.log",
  "primus_turbo_import_path": "<absolute_path_under_repo_root>",
  "kernel_source_snapshotted": true_or_false,
  "notes": "<short>"
}}
```

`primus_turbo_install_log` always points at the log file from step 4,
since the install is unconditional. `primus_turbo_import_path` must be
the value printed by the `python3 -P -c ...` verification in step 4c,
so later phases can cross-check which tree is live without being fooled
by the repo-root cwd shadow.

You MUST NOT modify the kernel source file itself in this phase. Only
snapshot it and scaffold the campaign directory. No chat output.
