# primus-turbo-auto-optimizer

把 `agent_workspace/Primus-Turbo/agent/skills/kernel-optimize/SKILL.md` 里定义的
kernel 优化循环封装成一条可长时运行的命令行：`primus-turbo-optimize`。

核心行为：
- 一条 prompt 启动一个 campaign，只在 DEFINE_TARGET 之后需要一次 `manifest.yaml` 确认，其余全程自动运行。
- DEFINE_TARGET → PREPARE_ENVIRONMENT → SURVEY_RELATED_WORK → READ_HISTORICAL_TIPS → BASELINE → (ANALYZE → OPTIMIZE → VALIDATE → DECIDE)\* → STAGNATION_REVIEW → TERMINATION_CHECK → REPORT，每个 phase 使用独立 `ClaudeSDKClient` 会话。
- 历史、tips、验证能力通过进程内 MCP server（`build_in_process_server`）暴露给 Claude；跨 campaign 的 `tips.md` 由 REPORT 末尾统一蒸馏写入，其他 phase 只读。
- SIGINT 两次语义：第一次触发 graceful stop（把当前 phase 跑完后直接跳到 REPORT），第二次立即退出。
- 所有状态落在 `state/<campaign_id>/run.json` 与 `state/<campaign_id>/phase_result/*.json`（按 campaign 分级互相隔离），中断后用 `-s <campaign_id>` 续跑。

详细架构见 `docs/kernel-optimize-cli-design.md`。

## 目录结构

```
turbo_optimize/
  cli.py                 argparse 入口 + CampaignParams 构造
  config.py              CampaignParams / campaign_id 规则
  state.py               run.json + phase_result/*.json 读写
  signals.py             SIGINT 双击语义
  skills.py              从 SKILL.md / optimize-loop.md 抽取节段
  manifest.py            manifest.yaml 读写 / y-e-n 确认 / 非 tty sentinel
  logs.py                logs/optimize.md & performance_trend.md 追加 + 解析
  scoring.py             benchmark CSV 解析 / geomean / accept-rollback 决策
  mcp/                   history / tips / verification 的进程内 MCP 工具
  prompts/*.md           每个 phase 的用户 prompt 模板
  orchestrator/
    run_phase.py         ClaudeAgentOptions 统一构造 + 流式执行
    phases/*.py          每个 phase 的 run() 协程
    campaign.py          主状态机、轮次执行、决策落盘、termination 检查
agent_workspace/Primus-Turbo/    Primus-Turbo 工作目录（由 scripts/sync_primus_turbo.sh 拉取最新 main，本仓库不跟踪）
scripts/sync_primus_turbo.sh     从 git@github.com:AMD-AGI/Primus-Turbo.git 克隆或 reset 到 origin/<branch>
docs/                            设计文档
tests/                           pytest 冒烟测试
```

## 环境要求

- Python 3.10+
- 能 clone `git@github.com:AMD-AGI/Primus-Turbo.git` 的 SSH 凭据（由 `scripts/sync_primus_turbo.sh` 使用）
- 如需真实跑优化循环：Claude 凭据（`ANTHROPIC_API_KEY` 或自建 gateway），以及 Primus-Turbo 所需的 GPU / ROCm / PyTorch / Triton 环境。冒烟测试本身不需要 Claude 凭据或 GPU。

## 安装

```bash
git clone <this-repo-url>
cd primus-turbo-auto-optimizer

bash scripts/sync_primus_turbo.sh

pip install -e '.[dev]'
```

`pip install -e '.[dev]'` 会把 `turbo_optimize` 以 editable 模式装进当前 Python 环境，
注册 `primus-turbo-optimize` 命令到 `PATH`，并安装 `claude-agent-sdk`、`pyyaml`、`pytest`。

校验：

```bash
primus-turbo-optimize --version
primus-turbo-optimize --help
```

### 同步 Primus-Turbo 源码

`agent_workspace/Primus-Turbo/` 不是 git submodule，而是由 `scripts/sync_primus_turbo.sh` 独立维护的克隆目录；本仓库在 `.gitignore` 里忽略它，因此不会把 Primus-Turbo 的 commit SHA 固化到父仓库。

脚本行为：目录不存在时 `git clone --branch <BRANCH>`；已存在时 `git fetch` + `git checkout <BRANCH>` + `git reset --hard origin/<BRANCH>`。默认分支 `main`，可通过环境变量覆盖：

```bash
# 默认：拉 origin/main 最新
bash scripts/sync_primus_turbo.sh

# 切换到其他分支（举一个例子，联调 dev/agent）
PRIMUS_TURBO_BRANCH=dev/agent bash scripts/sync_primus_turbo.sh

# 改用别的 URL（例如 fork / 镜像）
PRIMUS_TURBO_URL=git@github.com:<your-fork>/Primus-Turbo.git \
  bash scripts/sync_primus_turbo.sh
```

想象下面的使用场景帮助理解：每次进入容器重建环境，先跑一次 sync 脚本就能拿到 Primus-Turbo 最新主分支，父仓库无需任何 `git submodule` 操作，也不会再出现 "upload-pack: not our ref" 这类子模块 SHA 丢失的问题。

注意 `reset --hard` 会覆盖 `agent_workspace/Primus-Turbo/` 里任何本地未提交改动，本地有 WIP 请先在 Primus-Turbo 目录里 `git stash` 或换分支保存。

### 关于 venv

容器场景默认 **不要** 创建 venv，直接装到容器自带 Python 即可。原因：
- Primus-Turbo 的 `torch` / `triton` / ROCm 绑定通常预装在容器系统 `site-packages`，
  默认 venv 不继承系统包，BASELINE / VALIDATE 会立刻 `ImportError: torch`。
- 容器本身就是隔离边界，再套一层 venv 只是多一层 `PATH` 与解释器切换，没有收益。

以下两种情况才考虑 venv：
- 宿主机裸装 Python，且 Debian/Ubuntu 下 `pip install` 被 PEP 668
  `externally-managed-environment` 拒绝时：
  ```bash
  python3 -m venv --system-site-packages .venv
  source .venv/bin/activate
  pip install -e '.[dev]'
  ```
  `--system-site-packages` 保留宿主已装的 `torch` 等重型依赖。
- 同机同时维护多套互不兼容的 Python 依赖集合。

## 测试

仓库提供两条 pytest 冒烟测试，不会真正连 Claude / 跑 GPU。

```bash
pytest -q tests/test_smoke_orchestrator.py
```

两条用例分别覆盖：

1. `test_dry_run_plan`
   执行 `primus-turbo-optimize --dry-run` 等价逻辑，验证阶段规划打印完整、CampaignDir 被正确创建。

2. `test_smoke_campaign_reaches_done`
   把每个 phase 的 `run()` monkey-patch 成写出真实产物的桩函数（manifest.yaml / summary.md / phase_result JSON），再驱动完整主循环 `DEFINE_TARGET → DONE`。断言：
   - `state/<campaign_id>/run.json.current_phase == DONE` 且 `best_round` 被设置
   - `manifest.yaml` 存在
   - `logs/optimize.md` 包含 `Baseline` 与 `Termination Check` 小节
   - `logs/performance_trend.md` 至少 3 行（BASELINE + 2 个 ACCEPTED）
   - `logs/cost.md` 存在，表头包含 `Cost USD | Cumulative USD`
   - `rounds/round-1/summary.md` 与 `rounds/round-2/summary.md` 存在

手动再跑一次 dry-run 以确认 CLI 接线：

```bash
mkdir -p /tmp/turbo_dry/ws/agent
primus-turbo-optimize \
    -p "dry run check" \
    --workspace-root /tmp/turbo_dry/ws \
    --state-dir /tmp/turbo_dry/state \
    --dry-run
```

预期输出包含 `=== primus-turbo-optimize dry-run plan ===` 与 `DEFINE_TARGET / BASELINE / REPORT` 等 phase 名。

## 真实运行（需要 Claude 凭据 + Primus-Turbo 环境）

1. 配置 Claude 连接，二选一：

   ```bash
   export ANTHROPIC_API_KEY=sk-...
   # 或使用 gateway：
   export ANTHROPIC_BASE_URL=https://<your-gateway>/...
   export ANTHROPIC_AUTH_TOKEN=<gateway-token>
   ```

2. 启动一个新 campaign：

   ```bash
   primus-turbo-optimize -p "optimize gemm fp8 blockwise kernel with triton backend on MI300X"
   ```

   DEFINE_TARGET 完成后会在 `agent_workspace/Primus-Turbo/agent/workspace/<campaign_id>/manifest.yaml`
   生成草稿。TTY 环境下直接按 `y` 确认；非 TTY（如容器里后台运行）下，从另一个 shell 执行：

   ```bash
   touch agent_workspace/Primus-Turbo/agent/workspace/<campaign_id>/manifest.confirmed
   ```

### 热启脚本（`warm_restart.sh`）

campaign start 与 DEFINE_TARGET 结束后，orchestrator 会往 `<campaign_dir>/warm_restart.sh`
写一个 bash 包装器，把 `campaign_id`、`workspace_root`、`skills_root`、`state_dir`
的**绝对路径**固化进脚本。operator 不用记住目录也能从任意 cwd 恢复 campaign：

```bash
agent_workspace/Primus-Turbo/agent/workspace/<campaign_id>/warm_restart.sh -i 10
agent_workspace/Primus-Turbo/agent/workspace/<campaign_id>/warm_restart.sh -d 4h
agent_workspace/Primus-Turbo/agent/workspace/<campaign_id>/warm_restart.sh -i 5 -d 2h -- --debug-retry 5 -v
```

**必须**显式传 `-i / --iterations` 或 `-d / --duration` 中至少一个：上次停机往往
正是因为 `max_iterations` / `max_duration` 已经触发 T3 / T4，如果沿用旧值
TERMINATION_CHECK 会在新一轮开工前就再次命中。

`--` 之后的参数原样透传给 `primus-turbo-optimize`，常见用法：

| 场景 | 命令 |
|---|---|
| 只追加 5 轮 | `warm_restart.sh -i 5` |
| 追加时间预算 2h | `warm_restart.sh -d 2h` |
| 同时限定 10 轮与 4h | `warm_restart.sh -i 10 -d 4h` |
| 透传 debug-retry / verbose | `warm_restart.sh -i 5 -- --debug-retry 5 -v` |

脚本用 `shlex.quote` 对 campaign_id / 路径做 shell 转义，对带空格或特殊字符的目录也安全。

### 工作目录清洁规则（`workspace_hygiene`）

BASELINE / OPTIMIZE / VALIDATE / PREPARE_ENVIRONMENT 四个阶段的 prompt 都会注入一段
`<workspace_hygiene>` 块，硬性规定：

- 所有新建文件必须位于 `<campaign_dir>/` 下；不得写入 `<workspace_root>/` 根目录
  （例如 `agent_workspace/Primus-Turbo/`）
- 对项目源码的 in-place 编辑允许（`primus_turbo/` / `benchmarks/` / `tests/`），但不得在它们旁边新建文件
- benchmark / test 命令若把 CSV / log 默认输出到 cwd，必须用 `mv`（不是 `cp`）挪到
  `rounds/round-N/artifacts/`；用 `cp` 会在 repo 根目录留副本
- 往 `kernel_snapshot/` 拷贝时目标路径末尾必须带 `/`，避免 `cp` 把 dest 当文件名
- 阶段收尾前执行 `ls <workspace_root>`，核对没有新增顶层文件

规则块的真实源在 `turbo_optimize/skills.py:render_workspace_hygiene()`。

### Stray 文件手动清理（`--cleanup-stray`）

如果历史 campaign 留下了 stray 文件（例如 Claude 用 `cp` 而非 `mv` 把 benchmark CSV 复制到 repo 根目录），用这个命令把它们搬到 `<campaign_dir>/_stray/<timestamp>/`：

```bash
primus-turbo-optimize --cleanup-stray <campaign_id>              # dry-run 预览
primus-turbo-optimize --cleanup-stray <campaign_id> --apply      # 实际搬运
```

判定逻辑：在 `workspace_root` 里跑 `git status --porcelain`，挑出 `??` 开头的**顶层未跟踪文件**（忽略目录与子路径），逐个 `mv` 到 `<campaign_dir>/_stray/<timestamp>/`。
文件并不会被删除，保留在 `_stray/` 下可溯源。

### 模型 / 推理深度

两个旋钮全部由 CLI 或保存的 `run.json` 控制，不读环境变量，避免 shell
里误设的 `ANTHROPIC_MODEL` 悄悄改变某一个 campaign 的模型口径。

| 旋钮 | CLI | 内置 fallback |
|---|---|---|
| model | `--model` | `claude-opus-4-7` |
| effort | `--effort` | `max` |

`--effort` 合法取值 `low / medium / high / max`。

优先级（高者优先）：

1. CLI flag `--model` / `--effort`
2. `-s <campaign>` 续跑时 `state/run.json` 里保存的值
3. 内置 fallback

```bash
primus-turbo-optimize -p "..."                               # 用内置 fallback
primus-turbo-optimize -p "..." --model claude-sonnet-4-5     # CLI 覆盖 fallback
primus-turbo-optimize -s gemm_fp8_...                        # 沿用上次 run.json
primus-turbo-optimize -s gemm_fp8_... --effort high          # CLI 覆盖 run.json 并写回
```

### 调试重试（`--debug-retry`）

OPTIMIZE 的第一次实现可能带 build 错误或 correctness 失败（`ImportError`、
PTX 编译错、dtype 不一致、shape gating 漏 transpose 分支等）。此时 hypothesis
本身通常没问题，只是实现有 bug。`--debug-retry N` 打开一个 OPTIMIZE+VALIDATE
微循环：

- 默认 `N=3`，即每个 round 最多 4 次 OPTIMIZE+VALIDATE（1 次原始 + 3 次 retry）。
- 只有 `build failed` / `correctness failed` / `benchmark Check=FAIL` 三种
  ROLLBACK 原因会触发 retry。score 回退或"没改进"这类说明 hypothesis 本身
  无效，直接 ROLLBACK 并写 verified_ineffective，不浪费 retry 额度。
- retry 时会往 OPTIMIZE prompt 注入一段 `<retry_context>`，包含 attempt 号、
  失败原因、build_log 路径、失败 shape 列表，要求 Claude 保留 hypothesis
  只修 bug。

关闭 retry：

```bash
primus-turbo-optimize -p "..." --debug-retry 0
```

### ACCEPT 前的 full validation gate

`validate.md` 里 quick 只负责先筛，真正的 ACCEPT 需要 full 验证通过。Python
端按这一约束实现：

1. VALIDATE 以 `validation_level=quick` 先跑（产物写到
   `state/phase_result/validate_round<N>_quick.json`）。
2. 如果 quick 结果走向 `ACCEPTED` / `ACCEPT_PENDING_NOISE`，orchestrator
   立刻以 `validation_level=full` 再跑一次（产物写到
   `..._full.json`），把 `test_command` + `benchmark_command` 跑全。
3. full 的结果是最终决策。如果 full 翻盘（例如 full 发现 quick 没覆盖的
   shape 出 Check=FAIL）：
   - `state.history` 写 `ROLLED BACK`
   - `logs/optimize.md` / `performance_trend.md` 记的是 full 的 aggregate
   - `verified_ineffective` 追加这条方向
4. 如果 quick 已经是 ROLLBACK，full 不会再跑，省一次 benchmark 开销。

quick 与 full 两次产物路径不同，`run_phase` 的缓存复用不会误命中（否则
第二次 full 会直接拿到第一次 quick 的 JSON）。

### 容器内 root 运行（IS_SANDBOX）

每个 phase 都用 `permission_mode="bypassPermissions"`，底层等价于
`claude --dangerously-skip-permissions`。Claude Code 原生会拒绝 root 下
启用这个标志，报错：

```
--dangerously-skip-permissions cannot be used with root/sudo privileges
for security reasons
```

官方解法是设 `IS_SANDBOX=1`（参见 anthropics/claude-code#9184、#3490、
#927）。`ClaudeCodeConnector` 在构造时会自动检测：

1. `options.permission_mode == "bypassPermissions"`；
2. `os.geteuid() == 0`；
3. 调用方未显式设置 `IS_SANDBOX`（或显式设成空字符串）。

满足三个条件时，注入 `IS_SANDBOX=1` 到子进程 env。显式 `export IS_SANDBOX=0`
会被尊重（用于在非真实 sandbox 的裸机 root 下强制保留原始保护）。

### Git 提交开关

默认关闭：ACCEPT 轮不会自动 `git commit`，manifest / 代码改动留在工作树里
由用户自己处理。

| 旋钮 | 开关 | 说明 |
|---|---|---|
| `git_commit` | `--git-commit` / `--no-git-commit` | 默认 `--no-git-commit`。CLI 优先级高于 manifest，也就是说即便 DEFINE_TARGET 写出的 `manifest.yaml` 里保留了 `git_commit: true`，Python 端仍按 CLI 值执行，避免被 Claude 默认值污染。 |

如果确实想恢复"每个 ACCEPT 自动 commit"的老行为：

```bash
primus-turbo-optimize -p "..." --git-commit
```

### `--max-iterations` / `--max-duration`

两者都是 CLI 权威值。DEFINE_TARGET prompt 里会把当前值作为 "AUTHORITATIVE
CLI overrides" 段落注入给 Claude，Claude 必须按这个值写 `manifest.yaml`，
不会用 skill 里示例的 `null`。举一个例子：

```bash
primus-turbo-optimize -p "opt gemm fp8 blockwise" --max-iterations 3
```

生成的 draft `manifest.yaml` 里 `max_iterations: 3`，运行期 `run.json` 也
同步为 3。`--max-iterations` 必须在 `(0, 120)` 区间，否则 CLI 退出并提示。

### 路径归一化与已完成 phase 的复用

Python 主进程与 Claude 子进程的 cwd 不同（Claude 的 cwd 固定为
`--workspace-root`），因此相对路径在两边的含义不一致。campaign 启动时会把
`--workspace-root` / `--skills-root` / `--state-dir` / `campaign_dir` 一次性
`Path.resolve()` 成绝对路径，再传入 prompt 模板，避免 Claude 把
`state/phase_result/<phase>.json` 写到 `<workspace_root>/state/...` 却让
Python 在 shell cwd 下查不到的崩溃。

#### state 按 campaign_id 分级

`--state-dir`（默认 `state/`）只是一个**父目录**。campaign 启动时
orchestrator 把 `params.state_dir` 重写为 `<state_dir>/<campaign_id>/`，
之后 `run.json` / `phase_result/*.json` 都写进这一层；`_namespace_state_dir`
幂等，`warm_restart.sh` 再把 namespace 后的路径传回 CLI 也不会嵌套两层。

好处：并发多个 campaign（或一个 campaign 还没跑完就起新的 `-p` campaign）
不会互相覆盖 `run.json`，resume 语义只认自己那一级。

历史迁移有两类，campaign 启动时自动完成；已存在同名文件时不覆盖：

1. 升级前的单一 `state/run.json` + `state/phase_result/*.json`：第一次 resume
   到同一个 `campaign_id` 时搬到 `state/<id>/` 下。`campaign_id` 不匹配的
   legacy 文件保持原位，避免误吞别的 campaign 的残留。
2. Claude cwd 误写出来的 `<workspace_root>/state/phase_result/*.json`：同样
   搬到 namespace 后的绝对 `state_dir`。

另外，`run_phase` 在检测到 `expected_output` 文件已经存在且能解析成 JSON 时
会直接复用，不再启动 Claude 会话。也就是说 `-s <campaign>` 续跑某个已经把
JSON 产物写盘但还没来得及 `advance_phase` 的 phase 时，不会再花一次钱重跑；
只有在产物缺失 / 损坏时才会重新调用 Claude。

3. 中断与续跑：
   - 第一次 `Ctrl+C`：当前 phase 跑完后直接跳到 REPORT 生成最终报告。
   - 第二次 `Ctrl+C`：立即退出，状态保留。
   - 续跑：`primus-turbo-optimize -s <campaign_id>`。若中断发生在 OPTIMIZE/VALIDATE/DECIDE，
     状态机会把 `current_phase` 回退到同一轮的 ANALYZE 重跑，`current_round` 不变。

## 主要产物位置

| 路径 | 说明 |
|---|---|
| `state/<campaign_id>/run.json` | 主状态，包含 current_phase / current_round / best_round 等。按 campaign_id 分级避免并发 campaign 互相覆盖 |
| `state/<campaign_id>/phase_result/<PHASE>.json` 或 `<PHASE>_round<N>.json` | 每个 phase 的结构化输出；VALIDATE 会额外带 `_quick` / `_full` 后缀 |
| `agent_workspace/Primus-Turbo/agent/workspace/<campaign_id>/manifest.yaml` | 单次用户确认点 |
| `.../logs/optimize.md` | 追加式主日志，含 Baseline / History / Directions / Verified Ineffective / Final Report / Termination Check |
| `.../logs/performance_trend.md` | 每轮一行的性能趋势表 |
| `.../logs/cost.md` | 追加式成本账本，每次 `run_phase` 调用一行（含 cache 复用命中行，状态列区分 `ok` / `cached` / `interrupted` / `error:*`），最后一列是 `Cumulative USD`；resume 时自动从最后一行续写 |
| `.../rounds/round-<N>/summary.md` | 单轮摘要（SKILL 模板） |
| `.../rounds/round-<N>/kernel_snapshot/` | 该轮结束时的 kernel 源码快照 |
| `.../profiles/_transcript_<phase>.jsonl` | 每个 phase 的 Claude 会话 transcript |
| `<workspace_root>/agent/historical_experience/<gpu>/<op>/<backend>/tips.md` | 跨 campaign 知识库，只在 REPORT 末尾蒸馏写入，其他 phase 只读。用 `fcntl` 文件锁序列化并发 append |

### 历史经验 tips 的蒸馏规则

`tips.md` 的用途是下一个 campaign 启动时（可能在不同 GPU 或不同算子上）读到前人的失败原因与可迁移的成功模式，避免重复试错。因此 REPORT 阶段的 prompt 对 Claude 调用 `mcp__turbo__append_tip` 有硬质量闸门：

- 每次 campaign 最多 5 条，可以为 0，分 Failure / Success 两类。
- Failure tip 仅记录「有 profiler / benchmark 信号支撑的系统性失败」，编译错、拼写错、shape 对错这类本地 bug 不计入。
- Success tip 仅记录「与本次具体 shape / kernel path 无关、可在同 (backend, gpu) 下迁移到别的算子」的模式；纯 shape-specific 的 magic number 写进 `summary.md`，不写进 tips。
- 每条 tip 必须带四个字段：`context`（硬件 + 算子类 + backend 版本）、`signal`（可复测的 metric / profiler 计数 / error pattern）、`takeaway`（一句话、当作规则或约束的可复用陈述）、`applicability`（什么时候用 + 什么时候不要用）；任意字段缺失 Claude 都会被要求重写。

写入路径固定为 `<workspace_root>/agent/historical_experience/<target_gpu>/<target_op>/<target_backend>/tips.md`，并发 campaign 通过 `fcntl.flock` 串行化追加。`READ_HISTORICAL_TIPS` 在新 campaign 启动前读这张表，形成闭环。
