# primus-turbo-auto-optimizer

Two CLI tools that wrap the `kernel-optimize` skill defined in
`agent_workspace/Primus-Turbo/agent/skills/kernel-optimize/SKILL.md` into
a long-running, mostly unattended workflow:

- **`primus-turbo-optimize`** — drives the kernel-optimize loop end-to-end.
  A single user prompt opens a campaign, the orchestrator runs DEFINE_TARGET
  → PREPARE_ENVIRONMENT → SURVEY → READ_HISTORICAL_TIPS → BASELINE → (PROFILE
  → ANALYZE → OPTIMIZE → VALIDATE → REVIEW → DECIDE)\* → STAGNATION_REVIEW →
  TERMINATION_CHECK → REPORT, with one human checkpoint after DEFINE_TARGET
  to confirm `manifest.yaml`. Each phase runs in its own `ClaudeSDKClient`
  session; ACCEPT/ROLLBACK decisions are made in Python from parsed
  benchmark CSVs; campaign state is durable, so the run survives `Ctrl+C`
  and resumes via `-s <campaign_id>`.

- **`primus-turbo-view`** — renders one or many campaigns into a
  self-contained HTML dashboard (perf trend, cost, kernel-dispatch
  heatmaps, rocprof comparisons, live transcript tail). Output is a
  static `index.html` + `data.json`; `--watch` adds a small HTTP server
  that re-renders and live-reloads in the browser when files change.

Design documents:

- [`docs/turbo_optimize_design.md`](docs/turbo_optimize_design.md) —
  campaign orchestrator, phase state machine, IPC contracts, MCP tools,
  scoring rules.
- [`docs/turbo_view_design.md`](docs/turbo_view_design.md) — discovery,
  IO layer, analytics, payload schema, watch server.
- [`docs/kernel-optimize-cli-design.md`](docs/kernel-optimize-cli-design.md) —
  original CLI design doc kept for historical reference.

## How to run

### Install

Requires Python 3.10+ and SSH access to
`git@github.com:AMD-AGI/Primus-Turbo.git`. Real campaigns additionally need
a Claude credential and a Primus-Turbo / ROCm / PyTorch / Triton runtime;
the smoke tests do not.

```bash
git clone <this-repo-url>
cd primus-turbo-auto-optimizer

bash scripts/sync_primus_turbo.sh      # pulls Primus-Turbo into agent_workspace/

pip install -e '.[dev,view]'
```

`pip install -e '.[dev,view]'` registers two `console_scripts` on `PATH`:
`primus-turbo-optimize` and `primus-turbo-view`. The `view` extra adds
the HTML rendering dependencies (`jinja2`, `markdown-it-py`, `bleach`,
`watchdog`); drop it if you only need the optimizer.

In containers that already ship `torch` / `triton` / ROCm in the system
`site-packages`, install directly into the container Python rather than
a fresh `venv`. A plain `venv` does not inherit system packages and
BASELINE / VALIDATE will fail with `ImportError: torch`. When PEP 668
forces `--break-system-packages`, prefer:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e '.[dev,view]'
```

Verify:

```bash
primus-turbo-optimize --version
primus-turbo-view --version
```

### Run the optimizer

Configure Claude credentials first (one of):

```bash
export ANTHROPIC_API_KEY=sk-...
# or, behind a gateway:
export ANTHROPIC_BASE_URL=https://<your-gateway>/...
export ANTHROPIC_AUTH_TOKEN=<gateway-token>
```

Start a fresh campaign:

```bash
primus-turbo-optimize -p "optimize gemm fp8 blockwise kernel with triton backend on MI300X"
```

After DEFINE_TARGET writes the draft `manifest.yaml`, confirm it:

- Interactive (TTY): press `y` (accept), `e` (edit in `$EDITOR`), or `n`
  (abort) at the prompt.
- Non-interactive (e.g. nohup in a container): from another shell, run

  ```bash
  touch agent_workspace/Primus-Turbo/agent/workspace/<campaign_id>/manifest.confirmed
  ```

Resume an interrupted campaign:

```bash
primus-turbo-optimize -s <campaign_id>
```

Two `Ctrl+C` semantics: the first signal lets the current phase finish
and jumps to REPORT; a second signal exits immediately, leaving state on
disk so `-s <campaign_id>` can pick up.

Useful flags (full list via `--help`):

| Flag | Purpose |
|---|---|
| `--max-iterations N` | Cap rounds (must be in `(0, 120)`). |
| `--max-duration 4h` / `90m` | Wall-clock cap. |
| `--debug-retry N` | Retry budget for build / correctness bugs (default 3, `0` disables). |
| `--model claude-opus-4-7` | Override the model id. |
| `--effort low\|medium\|high\|max` | Override extended-thinking depth. |
| `--base-branch main` | Override `base_branch` in the manifest. |
| `--dry-run` | Print the phase plan without contacting Claude. |
| `--cleanup-stray <id> [--apply]` | Move stray top-level files from a previous run into `<campaign_dir>/_stray/`. |

Resume after the campaign already hit `DONE` and the budget needs to
grow:

```bash
agent_workspace/Primus-Turbo/agent/workspace/<campaign_id>/warm_restart.sh -i 10
agent_workspace/Primus-Turbo/agent/workspace/<campaign_id>/warm_restart.sh -d 4h
agent_workspace/Primus-Turbo/agent/workspace/<campaign_id>/warm_restart.sh -i 5 -- --debug-retry 5 -v
```

The script captures the absolute paths to `workspace_root`, `skills_root`,
`state_dir`, and `campaign_id`, so it can be invoked from any cwd.
At least one of `-i`/`--iterations` or `-d`/`--duration` is required;
otherwise `TERMINATION_CHECK` immediately re-fires whatever stopped the
previous run.

### Run the viewer

Render a single campaign to `<campaign_dir>/view/index.html`:

```bash
primus-turbo-view agent_workspace/Primus-Turbo/agent/workspace/<campaign_id>
```

Render all campaigns under a workspace root into one overview page:

```bash
primus-turbo-view agent_workspace/Primus-Turbo
```

Live mode (rebuilds on file changes, browser pushes a reload over SSE):

```bash
primus-turbo-view agent_workspace/Primus-Turbo/agent/workspace/<campaign_id> --watch
# or, force overview mode:
primus-turbo-view agent_workspace/Primus-Turbo --watch --multi
```

Default `--watch` host/port is `127.0.0.1:8765`; override with `--host`,
`--port`, or pass `--no-open` to skip the browser launch.

### Tests

Two pytest packages are shipped:

```bash
pytest -q tests/optimizer
pytest -q tests/view
```

Neither suite contacts Claude or requires a GPU. The optimizer smoke
test monkey-patches every phase to write deterministic phase outputs
and walks the orchestrator from `DEFINE_TARGET` to `DONE`.
