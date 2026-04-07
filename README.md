# Primus Optimizer

Claude Code plugin for automated multi-agent operator optimization in [Primus-Turbo](https://github.com/AMD-AI/Primus-Turbo).

Inspired by NVIDIA's AVO (arXiv:2603.24517), Primus Optimizer orchestrates parallel Worker agents — each optimizing a single operator x backend combination on an exclusive GPU — with automated profiling, bottleneck escalation, cross-pollination review, and web knowledge mining.

## Documentation

| Document | English | Chinese |
|----------|---------|---------|
| Design Specification | [design-spec.md](docs/design-spec.md) | [design-spec-zh.md](docs/design-spec-zh.md) |
| Implementation Plan | [implementation-plan.md](docs/implementation-plan.md) | [implementation-plan-zh.md](docs/implementation-plan-zh.md) |

## Architecture

```
User: /optimize --tasks gemm:triton,attn:ck --hw mi355x

                    ┌─────────────────────────────────────────────┐
                    │            Coordinator (optimize skill)      │
                    │  1. Parse config -> task matrix              │
                    │  2. Check GPU pool -> allocate GPUs          │
                    │  3. Dispatch Worker Agents (background,      │
                    │     worktree-isolated)                       │
                    │  4. Monitor loop -> update Dashboard         │
                    │  5. On bottleneck/completion -> Review Agent │
                    │  6. Finalize -> merge plan + PR descriptions │
                    └──────────────┬───────────────────────────────┘
               ┌───────────────────┼───────────────────┐
               ▼                   ▼                   ▼
        ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
        │ Worker Agent │     │ Worker Agent │     │ Worker Agent │
        │ gemm:triton  │     │ gemm:ck      │     │ attn:triton  │
        │ GPU:0        │     │ GPU:1        │     │ GPU:2        │
        │ worktree:A   │     │ worktree:B   │     │ worktree:C   │
        └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
               │                   │                   │
               ▼                   ▼                   ▼
        Profile -> Analyze -> Implement -> Verify -> Log (loop)
```

## Prerequisites

- Claude Code CLI installed and authenticated
- Primus-Turbo repository cloned with GPU access
- Python 3.10+ with `rich` and `pyyaml` installed
- AMD GPU with ROCm and HIP toolkit
- `rocprof` and `omniperf` available for profiling (optional)

## Installation

### 1. Clone the repository

```bash
# Clone alongside Primus-Turbo
cd /path/to/your/workspace
git clone <repo-url> primus-optimizer
```

### 2. Install Python dependencies

```bash
pip install rich pyyaml
```

### 3. Register skills in Claude Code

**Option A: Symlink (recommended)**

```bash
# From the primus-optimizer directory
ln -s $(pwd)/skills/optimize ~/.claude/skills/optimize
ln -s $(pwd)/skills/optimize-worker ~/.claude/skills/optimize-worker
ln -s $(pwd)/skills/optimize-review ~/.claude/skills/optimize-review
ln -s $(pwd)/skills/optimize-knowledge ~/.claude/skills/optimize-knowledge
```

**Option B: Copy**

```bash
cp -r skills/* ~/.claude/skills/
```

### 4. Configure GPU pool

Edit `config/optimizer.yaml` to match your GPU setup:

```yaml
gpu_pool:
  device_type: hip
  device_ids: [0, 1, 2, 3]   # Adjust to your available GPUs
```

### 5. Verify installation

```bash
# Run tests
cd primus-optimizer
python -m pytest tests/ -v

# Verify GPU availability
python -c "import subprocess; print(subprocess.run(['rocm-smi', '--showid'], capture_output=True, text=True).stdout)"
```

## Quick Start

**Single operator optimization:**
```bash
# In Claude Code
/optimize --tasks gemm:triton --hw mi355x
```

**Parallel optimization on multiple targets:**
```bash
/optimize --tasks gemm:ck,attention:triton --hw mi355x
```

**Use a predefined profile:**
```bash
/optimize --profile mi355x-priority
```

**Full sweep with custom round limit:**
```bash
/optimize --operators gemm,attention,grouped_gemm \
          --backends triton,ck \
          --hw mi300x \
          --max-rounds 5
```

## Monitoring

When `/optimize` starts, the Coordinator prints a Dashboard attach command. Open a separate terminal:

```bash
python primus-optimizer/tools/dashboard.py --state agent_docs/mi355x/optimizer-state.json
```

```
┌─ Primus Optimizer ──────────────────────────────────────────────────────────┐
│ Session: 2026-04-07-001 │ HW: MI355X │ Elapsed: 02:34:17 │ GPUs: 4/4 busy │
├─────────────────────────────────────────────────────────────────────────────┤
│  Worker Status                                                              │
│  ┌──────────────────┬────────┬─────┬──────────┬───────────┬────────────┐   │
│  │ Task             │ Status │ GPU │ Round    │ TFLOPS    │ Gain       │   │
│  ├──────────────────┼────────┼─────┼──────────┼───────────┼────────────┤   │
│  │ gemm:triton      │ ▶ RUN  │  0  │ 5/10     │ 488→793   │ +62.5%     │   │
│  │ gemm:ck          │ ▶ RUN  │  1  │ 3/10     │ 429→512   │ +19.3%     │   │
│  │ attn:triton      │ ⚠ BTNK │  2  │ 7/8  L1  │ 301→452   │ +50.2%     │   │
│  │ grouped_gemm:ck  │ ◻ WAIT │  -  │ -        │ -         │ -          │   │
│  └──────────────────┴────────┴─────┴──────────┴───────────┴────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

The Dashboard is read-only and fully independent — closing it does not affect the optimization.

## Project Structure

```
primus-optimizer/
├── config/
│   └── optimizer.yaml           # GPU pool, profiles, defaults, knowledge search
├── tools/
│   ├── config_loader.py         # YAML config + CLI task parsing
│   ├── state_store.py           # JSON state persistence with file locking
│   ├── gpu_pool.py              # Exclusive GPU device allocation
│   ├── benchmark_runner.py      # GPU-isolated benchmark command generation
│   ├── bottleneck_detector.py   # Multi-dimensional bottleneck detection
│   ├── profiler.py              # rocprof/omniperf command builders
│   ├── activity_logger.py       # Append-only JSONL activity logging
│   └── dashboard.py             # Rich terminal dashboard (standalone)
├── skills/
│   ├── optimize/SKILL.md        # Coordinator entry point
│   ├── optimize-worker/SKILL.md # Worker optimization loop
│   ├── optimize-review/SKILL.md # Review + cross-pollination
│   └── optimize-knowledge/SKILL.md # Web knowledge mining
├── templates/
│   ├── round-report.md          # Per-round report template
│   ├── pr-description.md        # PR description template
│   └── final-summary.md         # Session summary template
├── hooks/
│   ├── pre-benchmark.sh         # GPU environment check
│   └── post-round.sh            # Post-round file verification
├── tests/                       # 40 tests (pytest)
└── docs/
    ├── design-spec.md           # Design specification (EN)
    ├── design-spec-zh.md        # Design specification (ZH)
    ├── implementation-plan.md   # Implementation plan (EN)
    └── implementation-plan-zh.md # Implementation plan (ZH)
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No GPUs available" | Check `config/optimizer.yaml` gpu_pool device_ids |
| Worker stuck in VERIFY | Check `accuracy.log`; may need to relax SNR thresholds |
| Dashboard not updating | Verify `optimizer-state.json` is being written |
| Bottleneck L1 fails (no omniperf) | Install omniperf or skip to L2 (knowledge mining) |
| Worktree conflicts on merge | Follow `merge-plan.md` merge order |
| Session resume fails | Delete stale lock: `rm agent_docs/{hw}/optimizer-state.lock` |

## License

Internal use only.
