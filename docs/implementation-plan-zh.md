# Primus Optimizer 实现计划

> **面向 Agent 工作者：** 必须使用的子技能：superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans，按任务逐步实施本计划。各步骤使用复选框（`- [ ]`）语法进行跟踪。

**目标：** 构建 `primus-optimizer` Claude Code 插件，编排并行多 Agent 算子优化循环，服务于 Primus-Turbo 项目。

**架构：** 一个 Claude Code 技能插件，包含 4 个技能（coordinator、worker、review、knowledge）、6 个 Python 工具（state_store、gpu_pool、benchmark_runner、bottleneck_detector、profiler、dashboard），以及配套的配置/模板/钩子。技能是经过提示工程编写的 Markdown 文件，用于指导 Claude Code Agent；工具是 Agent 通过 Bash 调用的 Python 脚本。

**技术栈：** Python 3.10+、rich（终端仪表盘）、PyYAML（配置）、pytest（测试）。无外部服务依赖。

**规格文档：** `docs/superpowers/specs/2026-04-07-primus-optimizer-design.md`

---

## 文件结构

```
primus-optimizer/
├── tools/
│   ├── __init__.py
│   ├── state_store.py          # SessionState 读/写/更新（JSON 持久化）
│   ├── gpu_pool.py             # GPUPool 获取/释放，基于文件锁协调
│   ├── config_loader.py        # 加载 optimizer.yaml + CLI 参数解析
│   ├── benchmark_runner.py     # 封装现有 bench_*_turbo.py，实现 GPU 隔离
│   ├── bottleneck_detector.py  # 多维度瓶颈检测
│   ├── profiler.py             # rocprof/omniperf 命令生成
│   ├── activity_logger.py      # 仅追加的 JSONL 活动日志
│   └── dashboard.py            # Rich 终端仪表盘（独立运行）
├── skills/
│   ├── optimize/SKILL.md           # Coordinator 入口
│   ├── optimize-worker/SKILL.md    # Worker 优化循环
│   ├── optimize-review/SKILL.md    # Review + 交叉传播
│   └── optimize-knowledge/SKILL.md # Web 知识挖掘
├── templates/
│   ├── round-report.md
│   ├── pr-description.md
│   └── final-summary.md
├── hooks/
│   ├── pre-benchmark.sh
│   └── post-round.sh
├── config/
│   └── optimizer.yaml
└── tests/
    ├── __init__.py
    ├── test_state_store.py
    ├── test_gpu_pool.py
    ├── test_config_loader.py
    ├── test_benchmark_runner.py
    ├── test_bottleneck_detector.py
    ├── test_activity_logger.py
    └── test_dashboard.py
```

---

## 任务 1：项目脚手架 + 配置

**文件：**
- 创建：`primus-optimizer/tools/__init__.py`
- 创建：`primus-optimizer/tools/config_loader.py`
- 创建：`primus-optimizer/config/optimizer.yaml`
- 创建：`primus-optimizer/tests/__init__.py`
- 创建：`primus-optimizer/tests/test_config_loader.py`

- [ ] **步骤 1：创建目录结构**

```bash
mkdir -p primus-optimizer/{tools,skills/{optimize,optimize-worker,optimize-review,optimize-knowledge},templates,hooks,config,tests}
touch primus-optimizer/tools/__init__.py primus-optimizer/tests/__init__.py
```

- [ ] **步骤 2：编写默认 optimizer.yaml**

创建 `primus-optimizer/config/optimizer.yaml`：

```yaml
gpu_pool:
  device_type: hip
  device_ids: [0, 1, 2, 3]

defaults:
  max_rounds: 10
  bottleneck_threshold: 0.02
  bottleneck_patience: 3
  accuracy_snr_bf16: 30
  accuracy_snr_fp8: 20
  monitor_interval: 60

profiles:
  mi355x-priority:
    hw: mi355x
    tasks:
      - operator: gemm
        backend: ck
        max_rounds: 10
      - operator: attention
        backend: triton
        max_rounds: 8

  mi300x-full-sweep:
    hw: mi300x
    operators: [gemm, grouped_gemm, attention]
    backends: [triton, ck, hipblaslt]
    max_rounds: 5

  blockwise-fp8:
    hw: mi355x
    tasks:
      - operator: gemm
        backend: triton
        dtype: fp8
        quant: blockwise
        max_rounds: 12

knowledge:
  search_templates:
    gemm:
      queries:
        - "{backend} GEMM optimization AMD {arch} site:github.com"
        - "matrix multiplication kernel optimization rocm 2026"
        - "GEMM persistent kernel split-k AMD"
      repos_to_check:
        - ROCm/composable_kernel
        - ROCm/aiter
        - ROCm/hipBLASLt
        - triton-lang/triton
    attention:
      queries:
        - "flash attention optimization AMD MI300 MI350"
        - "triton attention kernel CDNA"
      repos_to_check:
        - Dao-AILab/flash-attention
        - ROCm/aiter
        - linkedin/Liger-Kernel
    grouped_gemm:
      queries:
        - "grouped GEMM MoE optimization AMD"
        - "batched matrix multiplication variable sizes GPU"
      repos_to_check:
        - ROCm/composable_kernel
        - vllm-project/vllm
```

- [ ] **步骤 3：编写 config_loader 的失败测试**

创建 `primus-optimizer/tests/test_config_loader.py`：

```python
import pytest
import os
import tempfile
import yaml
from tools.config_loader import load_config, parse_tasks, resolve_profile


def test_load_config_from_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"defaults": {"max_rounds": 5}, "gpu_pool": {"device_ids": [0, 1]}}, f)
        f.flush()
        cfg = load_config(f.name)
    os.unlink(f.name)
    assert cfg["defaults"]["max_rounds"] == 5
    assert cfg["gpu_pool"]["device_ids"] == [0, 1]


def test_parse_tasks_explicit():
    tasks = parse_tasks(tasks_str="gemm:ck,attention:triton", operators=None, backends=None)
    assert tasks == [
        {"operator": "gemm", "backend": "ck"},
        {"operator": "attention", "backend": "triton"},
    ]


def test_parse_tasks_cartesian():
    tasks = parse_tasks(tasks_str=None, operators="gemm,attention", backends="triton,ck")
    assert len(tasks) == 4
    assert {"operator": "gemm", "backend": "triton"} in tasks
    assert {"operator": "attention", "backend": "ck"} in tasks


def test_parse_tasks_mixed():
    tasks = parse_tasks(tasks_str="gemm:ck", operators="attention", backends="triton")
    assert len(tasks) == 2
    assert {"operator": "gemm", "backend": "ck"} in tasks
    assert {"operator": "attention", "backend": "triton"} in tasks


def test_resolve_profile():
    cfg = {
        "profiles": {
            "test-profile": {
                "hw": "mi355x",
                "tasks": [{"operator": "gemm", "backend": "ck", "max_rounds": 10}],
            }
        },
        "defaults": {"max_rounds": 10},
    }
    hw, tasks = resolve_profile(cfg, "test-profile")
    assert hw == "mi355x"
    assert tasks[0]["operator"] == "gemm"


def test_resolve_profile_cartesian():
    cfg = {
        "profiles": {
            "sweep": {
                "hw": "mi300x",
                "operators": ["gemm", "attention"],
                "backends": ["triton", "ck"],
            }
        },
        "defaults": {"max_rounds": 5},
    }
    hw, tasks = resolve_profile(cfg, "sweep")
    assert hw == "mi300x"
    assert len(tasks) == 4
```

- [ ] **步骤 4：运行测试确认失败**

```bash
cd primus-optimizer && python -m pytest tests/test_config_loader.py -v
```
预期结果：失败 -- `ModuleNotFoundError: No module named 'tools.config_loader'`

- [ ] **步骤 5：实现 config_loader.py**

创建 `primus-optimizer/tools/config_loader.py`：

```python
"""Load optimizer configuration from YAML and parse CLI task specifications."""

from __future__ import annotations

import os
from itertools import product
from pathlib import Path
from typing import Any

import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "optimizer.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load YAML config, falling back to the default bundled config."""
    path = Path(path) if path else _DEFAULT_CONFIG_PATH
    with open(path) as f:
        return yaml.safe_load(f)


def parse_tasks(
    tasks_str: str | None = None,
    operators: str | None = None,
    backends: str | None = None,
) -> list[dict[str, str]]:
    """Parse task specifications into a list of {operator, backend} dicts.

    Supports explicit ('gemm:ck,attn:triton'), cartesian (operators x backends),
    or mixed (both).
    """
    result: list[dict[str, str]] = []

    # Explicit tasks
    if tasks_str:
        for item in tasks_str.split(","):
            op, be = item.strip().split(":")
            result.append({"operator": op, "backend": be})

    # Cartesian product
    if operators and backends:
        ops = [o.strip() for o in operators.split(",")]
        bes = [b.strip() for b in backends.split(",")]
        for op, be in product(ops, bes):
            entry = {"operator": op, "backend": be}
            if entry not in result:
                result.append(entry)

    return result


def resolve_profile(
    cfg: dict[str, Any], profile_name: str
) -> tuple[str, list[dict[str, str]]]:
    """Resolve a named profile into (hw, tasks) pair."""
    profile = cfg["profiles"][profile_name]
    hw = profile["hw"]

    if "tasks" in profile:
        return hw, profile["tasks"]

    # Cartesian product from operators x backends
    ops = profile["operators"]
    bes = profile["backends"]
    tasks = [{"operator": op, "backend": be} for op, be in product(ops, bes)]
    return hw, tasks
```

- [ ] **步骤 6：运行测试确认通过**

```bash
cd primus-optimizer && python -m pytest tests/test_config_loader.py -v
```
预期结果：全部 5 个测试通过

- [ ] **步骤 7：提交**

```bash
git add primus-optimizer/
git commit -m "feat(optimizer): scaffold project + config loader with tests"
```

---

## 任务 2：状态存储 -- 会话状态持久化

**文件：**
- 创建：`primus-optimizer/tools/state_store.py`
- 创建：`primus-optimizer/tests/test_state_store.py`

- [ ] **步骤 1：编写 state_store 的失败测试**

创建 `primus-optimizer/tests/test_state_store.py`：

```python
import json
import os
import tempfile
import pytest
from tools.state_store import StateStore, WorkerStatus


@pytest.fixture
def tmp_state_path():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


def test_init_creates_state(tmp_state_path):
    store = StateStore(tmp_state_path)
    store.init_session(
        session_id="test-001",
        hw="mi355x",
        tasks=[{"operator": "gemm", "backend": "triton"}],
        gpu_ids=[0, 1],
    )
    state = store.load()
    assert state["session_id"] == "test-001"
    assert state["config"]["hw"] == "mi355x"
    assert len(state["workers"]) == 0
    assert state["task_queue"] == ["gemm:triton"]


def test_add_worker(tmp_state_path):
    store = StateStore(tmp_state_path)
    store.init_session("s1", "mi355x", [{"operator": "gemm", "backend": "ck"}], [0])
    store.add_worker("gemm:ck", gpu_id=0, worktree_branch="opt/gemm-ck-001")
    state = store.load()
    assert "gemm:ck" in state["workers"]
    assert state["workers"]["gemm:ck"]["status"] == WorkerStatus.RUNNING
    assert state["workers"]["gemm:ck"]["gpu_id"] == 0


def test_record_round(tmp_state_path):
    store = StateStore(tmp_state_path)
    store.init_session("s1", "mi355x", [{"operator": "gemm", "backend": "triton"}], [0])
    store.add_worker("gemm:triton", 0, "opt/gemm-triton-001")
    store.record_round(
        worker_id="gemm:triton",
        round_num=1,
        strategy="persistent_kernel",
        baseline_tflops=488.1,
        result_tflops=614.2,
        status="success",
    )
    state = store.load()
    w = state["workers"]["gemm:triton"]
    assert w["current_round"] == 1
    assert len(w["rounds"]) == 1
    assert w["rounds"][0]["improvement_pct"] == pytest.approx(25.8, abs=0.1)


def test_update_worker_status(tmp_state_path):
    store = StateStore(tmp_state_path)
    store.init_session("s1", "mi355x", [{"operator": "gemm", "backend": "triton"}], [0])
    store.add_worker("gemm:triton", 0, "opt/gemm-triton-001")
    store.update_worker_status("gemm:triton", WorkerStatus.BOTTLENECK, bottleneck_level="L1")
    state = store.load()
    assert state["workers"]["gemm:triton"]["status"] == WorkerStatus.BOTTLENECK
    assert state["workers"]["gemm:triton"]["bottleneck"] == "L1"


def test_add_review_log(tmp_state_path):
    store = StateStore(tmp_state_path)
    store.init_session("s1", "mi355x", [], [0])
    store.add_review_log(trigger="milestone", worker="gemm:triton", round_num=5, decision="continue")
    state = store.load()
    assert len(state["review_log"]) == 1
    assert state["review_log"][0]["trigger"] == "milestone"


def test_pop_task_queue(tmp_state_path):
    store = StateStore(tmp_state_path)
    store.init_session("s1", "mi355x", [
        {"operator": "gemm", "backend": "triton"},
        {"operator": "gemm", "backend": "ck"},
    ], [0])
    task = store.pop_task_queue()
    assert task == "gemm:triton"
    task2 = store.pop_task_queue()
    assert task2 == "gemm:ck"
    task3 = store.pop_task_queue()
    assert task3 is None
```

- [ ] **步骤 2：运行测试确认失败**

```bash
cd primus-optimizer && python -m pytest tests/test_state_store.py -v
```
预期结果：失败 -- `ModuleNotFoundError`

- [ ] **步骤 3：实现 state_store.py**

创建 `primus-optimizer/tools/state_store.py`：

```python
"""Persistent JSON state store for optimizer sessions."""

from __future__ import annotations

import json
import fcntl
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class WorkerStatus:
    PENDING = "pending"
    RUNNING = "running"
    BOTTLENECK = "bottleneck"
    STUCK = "stuck"
    COMPLETED = "completed"
    FAILED = "failed"


class StateStore:
    """Read/write optimizer-state.json with file-lock safety."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _read(self) -> dict[str, Any]:
        with open(self.path) as f:
            return json.load(f)

    def _write(self, state: dict[str, Any]) -> None:
        state["updated_at"] = self._now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        tmp.rename(self.path)

    def _update(self, fn) -> dict[str, Any]:
        """Read-modify-write with file lock."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(".lock")
        with open(lock_path, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                state = self._read()
                fn(state)
                self._write(state)
                return state
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def load(self) -> dict[str, Any]:
        return self._read()

    def init_session(
        self,
        session_id: str,
        hw: str,
        tasks: list[dict[str, str]],
        gpu_ids: list[int],
    ) -> None:
        state = {
            "session_id": session_id,
            "config": {
                "hw": hw,
                "tasks": tasks,
            },
            "gpu_pool": {
                "total": gpu_ids,
                "allocated": {},
            },
            "workers": {},
            "task_queue": [f"{t['operator']}:{t['backend']}" for t in tasks],
            "review_log": [],
            "started_at": self._now(),
            "updated_at": self._now(),
        }
        self._write(state)

    def add_worker(self, worker_id: str, gpu_id: int, worktree_branch: str) -> None:
        def _add(state):
            state["workers"][worker_id] = {
                "status": WorkerStatus.RUNNING,
                "gpu_id": gpu_id,
                "worktree_branch": worktree_branch,
                "current_round": 0,
                "rounds": [],
                "bottleneck": None,
                "started_at": self._now(),
            }
            state["gpu_pool"]["allocated"][worker_id] = gpu_id
        self._update(_add)

    def record_round(
        self,
        worker_id: str,
        round_num: int,
        strategy: str,
        baseline_tflops: float,
        result_tflops: float,
        status: str,
    ) -> None:
        improvement = ((result_tflops - baseline_tflops) / baseline_tflops) * 100 if baseline_tflops > 0 else 0.0

        def _record(state):
            w = state["workers"][worker_id]
            w["current_round"] = round_num
            w["rounds"].append({
                "round": round_num,
                "strategy": strategy,
                "baseline_tflops": baseline_tflops,
                "result_tflops": result_tflops,
                "improvement_pct": round(improvement, 1),
                "status": status,
            })
        self._update(_record)

    def update_worker_status(
        self, worker_id: str, status: str, bottleneck_level: str | None = None
    ) -> None:
        def _update_status(state):
            state["workers"][worker_id]["status"] = status
            if bottleneck_level is not None:
                state["workers"][worker_id]["bottleneck"] = bottleneck_level
        self._update(_update_status)

    def add_review_log(
        self, trigger: str, worker: str, round_num: int, decision: str,
        cross_pollination: list | None = None,
    ) -> None:
        def _add(state):
            state["review_log"].append({
                "trigger": trigger,
                "worker": worker,
                "round": round_num,
                "decision": decision,
                "cross_pollination": cross_pollination or [],
                "timestamp": self._now(),
            })
        self._update(_add)

    def pop_task_queue(self) -> str | None:
        result = [None]
        def _pop(state):
            if state["task_queue"]:
                result[0] = state["task_queue"].pop(0)
        self._update(_pop)
        return result[0]
```

- [ ] **步骤 4：运行测试**

```bash
cd primus-optimizer && python -m pytest tests/test_state_store.py -v
```
预期结果：全部 6 个测试通过

- [ ] **步骤 5：提交**

```bash
git add primus-optimizer/tools/state_store.py primus-optimizer/tests/test_state_store.py
git commit -m "feat(optimizer): state store with JSON persistence and file locking"
```

---

## 任务 3：GPU 资源池管理器

**文件：**
- 创建：`primus-optimizer/tools/gpu_pool.py`
- 创建：`primus-optimizer/tests/test_gpu_pool.py`

- [ ] **步骤 1：编写失败测试**

创建 `primus-optimizer/tests/test_gpu_pool.py`：

```python
import pytest
from tools.gpu_pool import GPUPool


def test_acquire_returns_gpu():
    pool = GPUPool([0, 1, 2])
    gpu = pool.acquire("gemm:triton")
    assert gpu in [0, 1, 2]
    assert pool.available_count() == 2


def test_release_returns_gpu_to_pool():
    pool = GPUPool([0, 1])
    pool.acquire("task1")
    pool.release("task1")
    assert pool.available_count() == 2


def test_acquire_exhausted_raises():
    pool = GPUPool([0])
    pool.acquire("task1")
    with pytest.raises(RuntimeError, match="No GPUs available"):
        pool.acquire("task2")


def test_release_unknown_raises():
    pool = GPUPool([0])
    with pytest.raises(KeyError):
        pool.release("unknown")


def test_allocated_map():
    pool = GPUPool([0, 1])
    pool.acquire("task1")
    pool.acquire("task2")
    alloc = pool.allocated_map()
    assert len(alloc) == 2
    assert "task1" in alloc
    assert "task2" in alloc


def test_has_available():
    pool = GPUPool([0])
    assert pool.has_available() is True
    pool.acquire("t")
    assert pool.has_available() is False


def test_env_var():
    pool = GPUPool([3])
    gpu = pool.acquire("task1")
    assert pool.env_for("task1") == {"HIP_VISIBLE_DEVICES": "3"}
```

- [ ] **步骤 2：运行测试确认失败**

```bash
cd primus-optimizer && python -m pytest tests/test_gpu_pool.py -v
```

- [ ] **步骤 3：实现 gpu_pool.py**

创建 `primus-optimizer/tools/gpu_pool.py`：

```python
"""GPU resource pool for allocating exclusive devices to Workers."""

from __future__ import annotations


class GPUPool:
    def __init__(self, gpu_ids: list[int]):
        self._available: list[int] = list(gpu_ids)
        self._allocated: dict[str, int] = {}

    def acquire(self, task_id: str) -> int:
        if not self._available:
            raise RuntimeError("No GPUs available")
        gpu_id = self._available.pop(0)
        self._allocated[task_id] = gpu_id
        return gpu_id

    def release(self, task_id: str) -> None:
        gpu_id = self._allocated.pop(task_id)  # raises KeyError if unknown
        self._available.append(gpu_id)

    def has_available(self) -> bool:
        return len(self._available) > 0

    def available_count(self) -> int:
        return len(self._available)

    def allocated_map(self) -> dict[str, int]:
        return dict(self._allocated)

    def env_for(self, task_id: str) -> dict[str, str]:
        gpu_id = self._allocated[task_id]
        return {"HIP_VISIBLE_DEVICES": str(gpu_id)}
```

- [ ] **步骤 4：运行测试**

```bash
cd primus-optimizer && python -m pytest tests/test_gpu_pool.py -v
```
预期结果：全部 7 个测试通过

- [ ] **步骤 5：提交**

```bash
git add primus-optimizer/tools/gpu_pool.py primus-optimizer/tests/test_gpu_pool.py
git commit -m "feat(optimizer): GPU pool manager with exclusive device allocation"
```

---

## 任务 4：活动日志记录器

**文件：**
- 创建：`primus-optimizer/tools/activity_logger.py`
- 创建：`primus-optimizer/tests/test_activity_logger.py`

- [ ] **步骤 1：编写失败测试**

创建 `primus-optimizer/tests/test_activity_logger.py`：

```python
import json
import os
import tempfile
import pytest
from tools.activity_logger import ActivityLogger


@pytest.fixture
def tmp_log():
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


def test_log_appends_jsonl(tmp_log):
    logger = ActivityLogger(tmp_log, worker_id="gemm:triton")
    logger.log("VERIFY", 3, "accuracy check passed")
    logger.log("VERIFY", 3, "benchmark complete: 793.2 TFLOPS")

    with open(tmp_log) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == 2
    assert lines[0]["worker"] == "gemm:triton"
    assert lines[0]["phase"] == "VERIFY"
    assert lines[0]["round"] == 3
    assert "ts" in lines[0]


def test_read_recent(tmp_log):
    logger = ActivityLogger(tmp_log, worker_id="gemm:ck")
    for i in range(20):
        logger.log("PROFILE", i, f"msg {i}")
    entries = ActivityLogger.read_recent(tmp_log, n=5)
    assert len(entries) == 5
    assert entries[-1]["msg"] == "msg 19"
```

- [ ] **步骤 2：实现 activity_logger.py**

创建 `primus-optimizer/tools/activity_logger.py`：

```python
"""Append-only JSONL activity logger for Worker progress tracking."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class ActivityLogger:
    def __init__(self, path: str | Path, worker_id: str):
        self.path = Path(path)
        self.worker_id = worker_id
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, phase: str, round_num: int, msg: str) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "worker": self.worker_id,
            "phase": phase,
            "round": round_num,
            "msg": msg,
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    @staticmethod
    def read_recent(path: str | Path, n: int = 10) -> list[dict]:
        path = Path(path)
        if not path.exists():
            return []
        with open(path) as f:
            lines = f.readlines()
        recent = lines[-n:] if len(lines) > n else lines
        return [json.loads(line) for line in recent if line.strip()]
```

- [ ] **步骤 3：运行测试**

```bash
cd primus-optimizer && python -m pytest tests/test_activity_logger.py -v
```
预期结果：全部 2 个测试通过

- [ ] **步骤 4：提交**

```bash
git add primus-optimizer/tools/activity_logger.py primus-optimizer/tests/test_activity_logger.py
git commit -m "feat(optimizer): JSONL activity logger for worker progress tracking"
```

---

## 任务 5：基准测试运行器

**文件：**
- 创建：`primus-optimizer/tools/benchmark_runner.py`
- 创建：`primus-optimizer/tests/test_benchmark_runner.py`

- [ ] **步骤 1：编写失败测试**

创建 `primus-optimizer/tests/test_benchmark_runner.py`：

```python
import pytest
from tools.benchmark_runner import BenchmarkRunner


def test_build_benchmark_cmd_gemm():
    runner = BenchmarkRunner(repo_root="/repo", gpu_id=2)
    cmd = runner.build_benchmark_cmd(
        operator="gemm", dtype="fp8", granularity="blockwise", output_csv="/out/result.csv"
    )
    assert "HIP_VISIBLE_DEVICES=2" in cmd
    assert "benchmark/ops/bench_gemm_turbo.py" in cmd
    assert "--dtype fp8" in cmd
    assert "--granularity blockwise" in cmd
    assert "--output /out/result.csv" in cmd


def test_build_benchmark_cmd_grouped_gemm():
    runner = BenchmarkRunner(repo_root="/repo", gpu_id=0)
    cmd = runner.build_benchmark_cmd(operator="grouped_gemm", dtype="bf16")
    assert "bench_grouped_gemm_turbo.py" in cmd
    assert "--dtype bf16" in cmd


def test_build_benchmark_cmd_attention():
    runner = BenchmarkRunner(repo_root="/repo", gpu_id=1)
    cmd = runner.build_benchmark_cmd(operator="attention")
    assert "bench_attention_turbo.py" in cmd


def test_build_accuracy_cmd():
    runner = BenchmarkRunner(repo_root="/repo", gpu_id=3)
    cmd = runner.build_accuracy_cmd(operator="gemm", report_dir="/out/accuracy")
    assert "HIP_VISIBLE_DEVICES=3" in cmd
    assert "eval_gemm_accuracy.py" in cmd
    assert "--report-dir-path /out/accuracy" in cmd


def test_unknown_operator_raises():
    runner = BenchmarkRunner(repo_root="/repo", gpu_id=0)
    with pytest.raises(ValueError, match="Unknown operator"):
        runner.build_benchmark_cmd(operator="unknown_op")
```

- [ ] **步骤 2：实现 benchmark_runner.py**

创建 `primus-optimizer/tools/benchmark_runner.py`：

```python
"""Standardized benchmark command builder with GPU isolation."""

from __future__ import annotations

from pathlib import Path

_BENCH_SCRIPTS = {
    "gemm": "benchmark/ops/bench_gemm_turbo.py",
    "grouped_gemm": "benchmark/ops/bench_grouped_gemm_turbo.py",
    "attention": "benchmark/ops/bench_attention_turbo.py",
}

_ACCURACY_SCRIPTS = {
    "gemm": "benchmark/accuracy/eval_gemm_accuracy.py",
}


class BenchmarkRunner:
    def __init__(self, repo_root: str | Path, gpu_id: int):
        self.repo_root = Path(repo_root)
        self.gpu_id = gpu_id

    def _env_prefix(self) -> str:
        return f"HIP_VISIBLE_DEVICES={self.gpu_id}"

    def build_benchmark_cmd(
        self,
        operator: str,
        dtype: str = "bf16",
        granularity: str | None = None,
        output_csv: str | None = None,
    ) -> str:
        if operator not in _BENCH_SCRIPTS:
            raise ValueError(f"Unknown operator: {operator}. Valid: {list(_BENCH_SCRIPTS)}")

        script = self.repo_root / _BENCH_SCRIPTS[operator]
        parts = [self._env_prefix(), "python", str(script), f"--dtype {dtype}"]

        if granularity and dtype in ("fp8", "fp4"):
            parts.append(f"--granularity {granularity}")
        if output_csv:
            parts.append(f"--output {output_csv}")

        return " ".join(parts)

    def build_accuracy_cmd(
        self, operator: str, report_dir: str, seed: int = 42
    ) -> str:
        if operator not in _ACCURACY_SCRIPTS:
            raise ValueError(f"No accuracy script for operator: {operator}")

        script = self.repo_root / _ACCURACY_SCRIPTS[operator]
        return f"{self._env_prefix()} python {script} --report-dir-path {report_dir} --seed {seed}"
```

- [ ] **步骤 3：运行测试**

```bash
cd primus-optimizer && python -m pytest tests/test_benchmark_runner.py -v
```
预期结果：全部 5 个测试通过

- [ ] **步骤 4：提交**

```bash
git add primus-optimizer/tools/benchmark_runner.py primus-optimizer/tests/test_benchmark_runner.py
git commit -m "feat(optimizer): benchmark runner with GPU-isolated command generation"
```

---

## 任务 6：瓶颈检测器

**文件：**
- 创建：`primus-optimizer/tools/bottleneck_detector.py`
- 创建：`primus-optimizer/tests/test_bottleneck_detector.py`

- [ ] **步骤 1：编写失败测试**

创建 `primus-optimizer/tests/test_bottleneck_detector.py`：

```python
import pytest
from tools.bottleneck_detector import BottleneckDetector, BottleneckResult


def _make_rounds(gains: list[float]) -> list[dict]:
    """辅助函数：根据给定的提升百分比创建轮次字典。"""
    return [
        {"round": i + 1, "improvement_pct": g, "status": "success"}
        for i, g in enumerate(gains)
    ]


def test_no_bottleneck():
    det = BottleneckDetector(threshold=0.02, patience=3)
    rounds = _make_rounds([25.0, 15.0, 8.0])
    result = det.check(rounds)
    assert result.is_bottleneck is False


def test_diminishing_returns_triggers():
    det = BottleneckDetector(threshold=0.02, patience=3)
    rounds = _make_rounds([25.0, 15.0, 8.0, 1.5, 0.8, 1.2])
    result = det.check(rounds)
    assert result.is_bottleneck is True
    assert result.reason == "diminishing_returns"


def test_near_roofline():
    det = BottleneckDetector(threshold=0.02, patience=3)
    rounds = _make_rounds([25.0, 15.0])
    result = det.check(rounds, current_utilization=0.85)
    assert result.is_bottleneck is True
    assert result.reason == "near_roofline"


def test_near_sota():
    det = BottleneckDetector(threshold=0.02, patience=3)
    rounds = _make_rounds([25.0])
    result = det.check(rounds, current_tflops=950, sota_tflops=980)
    assert result.is_bottleneck is True
    assert result.reason == "near_sota"


def test_not_near_sota():
    det = BottleneckDetector(threshold=0.02, patience=3)
    rounds = _make_rounds([25.0])
    result = det.check(rounds, current_tflops=500, sota_tflops=980)
    assert result.is_bottleneck is False


def test_empty_rounds():
    det = BottleneckDetector(threshold=0.02, patience=3)
    result = det.check([])
    assert result.is_bottleneck is False
```

- [ ] **步骤 2：实现 bottleneck_detector.py**

创建 `primus-optimizer/tools/bottleneck_detector.py`：

```python
"""Multi-dimensional bottleneck detection for optimization loops."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BottleneckResult:
    is_bottleneck: bool
    reason: str | None = None  # "diminishing_returns" | "near_roofline" | "near_sota"
    details: str | None = None


class BottleneckDetector:
    def __init__(
        self,
        threshold: float = 0.02,
        patience: int = 3,
        roofline_ceiling: float = 0.80,
        sota_gap: float = 0.05,
    ):
        self.threshold = threshold
        self.patience = patience
        self.roofline_ceiling = roofline_ceiling
        self.sota_gap = sota_gap

    def check(
        self,
        rounds: list[dict],
        current_utilization: float | None = None,
        current_tflops: float | None = None,
        sota_tflops: float | None = None,
    ) -> BottleneckResult:
        # 优先检查 roofline（最高置信度信号）
        if current_utilization is not None and current_utilization > self.roofline_ceiling:
            return BottleneckResult(
                is_bottleneck=True,
                reason="near_roofline",
                details=f"Utilization {current_utilization:.1%} > {self.roofline_ceiling:.0%} ceiling",
            )

        # 检查 SOTA 差距
        if current_tflops is not None and sota_tflops is not None and sota_tflops > 0:
            gap = (sota_tflops - current_tflops) / sota_tflops
            if gap < self.sota_gap:
                return BottleneckResult(
                    is_bottleneck=True,
                    reason="near_sota",
                    details=f"Gap to SOTA: {gap:.1%} < {self.sota_gap:.0%} threshold",
                )

        # 检查收益递减
        if len(rounds) >= self.patience:
            recent = rounds[-self.patience:]
            all_low = all(
                abs(r.get("improvement_pct", 0)) / 100 < self.threshold
                for r in recent
                if r.get("status") == "success"
            )
            successful = [r for r in recent if r.get("status") == "success"]
            if len(successful) >= self.patience and all_low:
                return BottleneckResult(
                    is_bottleneck=True,
                    reason="diminishing_returns",
                    details=f"Last {self.patience} rounds all < {self.threshold:.0%} improvement",
                )

        return BottleneckResult(is_bottleneck=False)
```

- [ ] **步骤 3：运行测试**

```bash
cd primus-optimizer && python -m pytest tests/test_bottleneck_detector.py -v
```
预期结果：全部 6 个测试通过

- [ ] **步骤 4：提交**

```bash
git add primus-optimizer/tools/bottleneck_detector.py primus-optimizer/tests/test_bottleneck_detector.py
git commit -m "feat(optimizer): multi-dimensional bottleneck detector"
```

---

## 任务 7：性能分析命令构建器

**文件：**
- 创建：`primus-optimizer/tools/profiler.py`
- 创建：`primus-optimizer/tests/test_profiler.py`

- [ ] **步骤 1：编写失败测试**

创建 `primus-optimizer/tests/test_profiler.py`：

```python
import pytest
from tools.profiler import Profiler


def test_rocprof_cmd():
    p = Profiler(gpu_id=2)
    cmd = p.build_rocprof_cmd("python bench.py", output_dir="/out")
    assert "HIP_VISIBLE_DEVICES=2" in cmd
    assert "rocprof --stats" in cmd
    assert "-o /out/rocprof_stats.csv" in cmd
    assert "python bench.py" in cmd


def test_omniperf_cmd():
    p = Profiler(gpu_id=1)
    cmd = p.build_omniperf_cmd("python bench.py", output_dir="/out")
    assert "HIP_VISIBLE_DEVICES=1" in cmd
    assert "omniperf profile" in cmd
    assert "--path /out/omniperf" in cmd


def test_roofline_analysis_cmd():
    p = Profiler(gpu_id=0)
    cmd = p.build_roofline_cmd(omniperf_dir="/out/omniperf")
    assert "omniperf analyze" in cmd
    assert "--path /out/omniperf" in cmd
```

- [ ] **步骤 2：实现 profiler.py**

创建 `primus-optimizer/tools/profiler.py`：

```python
"""Profiling command builders for rocprof and omniperf."""

from __future__ import annotations


class Profiler:
    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id

    def _env_prefix(self) -> str:
        return f"HIP_VISIBLE_DEVICES={self.gpu_id}"

    def build_rocprof_cmd(self, bench_cmd: str, output_dir: str) -> str:
        return (
            f"{self._env_prefix()} rocprof --stats "
            f"-o {output_dir}/rocprof_stats.csv "
            f"{bench_cmd}"
        )

    def build_omniperf_cmd(self, bench_cmd: str, output_dir: str) -> str:
        return (
            f"{self._env_prefix()} omniperf profile "
            f"--path {output_dir}/omniperf "
            f"-- {bench_cmd}"
        )

    def build_roofline_cmd(self, omniperf_dir: str) -> str:
        return f"omniperf analyze --path {omniperf_dir} --roof-only"
```

- [ ] **步骤 3：运行测试**

```bash
cd primus-optimizer && python -m pytest tests/test_profiler.py -v
```
预期结果：全部 3 个测试通过

- [ ] **步骤 4：提交**

```bash
git add primus-optimizer/tools/profiler.py primus-optimizer/tests/test_profiler.py
git commit -m "feat(optimizer): profiler command builder for rocprof and omniperf"
```

---

## 任务 8：终端仪表盘

**文件：**
- 创建：`primus-optimizer/tools/dashboard.py`
- 创建：`primus-optimizer/tests/test_dashboard.py`

- [ ] **步骤 1：编写失败测试**

创建 `primus-optimizer/tests/test_dashboard.py`：

```python
import json
import os
import tempfile
import pytest
from tools.dashboard import (
    format_status_icon,
    format_worker_table_data,
    format_gain,
)


def test_format_status_icon():
    assert format_status_icon("running") == "▶ RUN"
    assert format_status_icon("bottleneck") == "⚠ BTNK"
    assert format_status_icon("completed") == "✓ DONE"
    assert format_status_icon("failed") == "✗ FAIL"
    assert format_status_icon("pending") == "◻ WAIT"


def test_format_gain():
    assert format_gain(62.5) == "+62.5%"
    assert format_gain(-3.2) == "-3.2%"
    assert format_gain(0.0) == "+0.0%"
    assert format_gain(None) == "-"


def test_format_worker_table_data():
    workers = {
        "gemm:triton": {
            "status": "running",
            "gpu_id": 0,
            "current_round": 5,
            "rounds": [
                {"round": 1, "baseline_tflops": 488.1, "result_tflops": 793.2, "improvement_pct": 62.5}
            ],
            "bottleneck": None,
        }
    }
    rows = format_worker_table_data(workers, max_rounds=10)
    assert len(rows) == 1
    row = rows[0]
    assert row["task"] == "gemm:triton"
    assert row["status"] == "▶ RUN"
    assert row["gpu"] == "0"
    assert row["round"] == "5/10"
    assert "488" in row["tflops"]
    assert "793" in row["tflops"]


def test_format_worker_table_pending():
    workers = {
        "gemm:ck": {
            "status": "pending",
            "gpu_id": None,
            "current_round": 0,
            "rounds": [],
            "bottleneck": None,
        }
    }
    rows = format_worker_table_data(workers, max_rounds=10)
    assert rows[0]["gpu"] == "-"
    assert rows[0]["round"] == "-"
```

- [ ] **步骤 2：实现 dashboard.py（格式化函数 + CLI 入口）**

创建 `primus-optimizer/tools/dashboard.py`：

```python
"""Terminal dashboard for monitoring optimizer state (standalone process)."""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
except ImportError:
    print("Dashboard requires 'rich'. Install: pip install rich")
    sys.exit(1)


# --- 格式化辅助函数（可独立测试） ---

_STATUS_ICONS = {
    "running": "▶ RUN",
    "bottleneck": "⚠ BTNK",
    "completed": "✓ DONE",
    "failed": "✗ FAIL",
    "pending": "◻ WAIT",
    "stuck": "✗ STUCK",
    "paused": "⏸ PAUSE",
}


def format_status_icon(status: str) -> str:
    return _STATUS_ICONS.get(status, status)


def format_gain(pct: float | None) -> str:
    if pct is None:
        return "-"
    return f"+{pct:.1f}%" if pct >= 0 else f"{pct:.1f}%"


def format_worker_table_data(
    workers: dict, max_rounds: int = 10
) -> list[dict[str, str]]:
    rows = []
    for task_id, w in workers.items():
        status_str = format_status_icon(w["status"])
        if w.get("bottleneck"):
            status_str += f" {w['bottleneck']}"

        gpu = str(w["gpu_id"]) if w.get("gpu_id") is not None else "-"

        if w["current_round"] > 0 or w["status"] == "running":
            mr = w.get("max_rounds", max_rounds)
            round_str = f"{w['current_round']}/{mr}"
        else:
            round_str = "-"

        if w["rounds"]:
            first_baseline = w["rounds"][0].get("baseline_tflops", 0)
            last_result = w["rounds"][-1].get("result_tflops", 0)
            tflops = f"{first_baseline:.0f}→{last_result:.0f}"
            total_gain = ((last_result - first_baseline) / first_baseline * 100) if first_baseline > 0 else 0
            gain = format_gain(total_gain)
        else:
            tflops = "-"
            gain = "-"

        rows.append({
            "task": task_id,
            "status": status_str,
            "gpu": gpu,
            "round": round_str,
            "tflops": tflops,
            "gain": gain,
        })
    return rows


# --- 仪表盘渲染 ---

def build_worker_table(rows: list[dict[str, str]]) -> Table:
    table = Table(title="Worker Status", expand=True)
    table.add_column("Task", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("GPU", justify="center")
    table.add_column("Round", justify="center")
    table.add_column("TFLOPS", justify="right")
    table.add_column("Gain", justify="right", style="green")

    for r in rows:
        style = None
        if "BTNK" in r["status"]:
            style = "yellow"
        elif "FAIL" in r["status"] or "STUCK" in r["status"]:
            style = "red"
        elif "DONE" in r["status"]:
            style = "green"
        table.add_row(r["task"], r["status"], r["gpu"], r["round"], r["tflops"], r["gain"], style=style)
    return table


def build_activity_panel(activity_entries: list[dict], max_lines: int = 8) -> Panel:
    lines = []
    for e in activity_entries[-max_lines:]:
        ts = e.get("ts", "")[:19].replace("T", " ")
        worker = e.get("worker", "?")
        msg = e.get("msg", "")
        lines.append(f"[dim]{ts}[/dim] [{worker:<16}] {msg}")
    content = "\n".join(lines) if lines else "[dim]No activity yet[/dim]"
    return Panel(content, title="Live Activity", border_style="blue")


def load_activities(pattern: str, n: int = 20) -> list[dict]:
    entries = []
    for path in glob.glob(pattern):
        try:
            with open(path) as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))
        except (json.JSONDecodeError, OSError):
            pass
    entries.sort(key=lambda e: e.get("ts", ""))
    return entries[-n:]


def run_dashboard(state_path: str, activity_pattern: str | None = None, interval: int = 5):
    console = Console()
    state_path = Path(state_path)

    if activity_pattern is None:
        activity_pattern = str(state_path.parent / "*" / "activity.jsonl")

    console.print(f"[bold]Primus Optimizer Dashboard[/bold]")
    console.print(f"Watching: {state_path}")
    console.print(f"Activity: {activity_pattern}")
    console.print("Press Ctrl+C to exit\n")

    try:
        with Live(console=console, refresh_per_second=1) as live:
            while True:
                if state_path.exists():
                    with open(state_path) as f:
                        state = json.load(f)

                    # 头部信息
                    sid = state.get("session_id", "?")
                    hw = state.get("config", {}).get("hw", "?")
                    started = state.get("started_at", "")
                    elapsed = ""
                    if started:
                        try:
                            start_dt = datetime.fromisoformat(started)
                            delta = datetime.now(timezone.utc) - start_dt
                            h, rem = divmod(int(delta.total_seconds()), 3600)
                            m, s = divmod(rem, 60)
                            elapsed = f"{h:02d}:{m:02d}:{s:02d}"
                        except ValueError:
                            elapsed = "?"

                    workers = state.get("workers", {})
                    gpu_total = len(state.get("gpu_pool", {}).get("total", []))
                    gpu_busy = len(state.get("gpu_pool", {}).get("allocated", {}))

                    header = Text(
                        f" Session: {sid} │ HW: {hw} │ Elapsed: {elapsed} │ GPUs: {gpu_busy}/{gpu_total} busy",
                        style="bold white on blue",
                    )

                    max_rounds = state.get("config", {}).get("defaults", {}).get("max_rounds", 10)
                    rows = format_worker_table_data(workers, max_rounds)
                    table = build_worker_table(rows)

                    activities = load_activities(activity_pattern)
                    activity_panel = build_activity_panel(activities)

                    layout = Layout()
                    layout.split_column(
                        Layout(header, size=1),
                        Layout(table, name="table"),
                        Layout(activity_panel, name="activity", size=12),
                    )
                    live.update(layout)
                else:
                    live.update(Text(f"Waiting for {state_path} ...", style="yellow"))

                time.sleep(interval)
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard closed.[/yellow]")


def main():
    parser = argparse.ArgumentParser(description="Primus Optimizer Dashboard")
    parser.add_argument("--state", required=True, help="Path to optimizer-state.json")
    parser.add_argument("--activity", default=None, help="Glob pattern for activity.jsonl files")
    parser.add_argument("--interval", type=int, default=5, help="Refresh interval in seconds")
    parser.add_argument("--keep-alive", action="store_true", help="Stay open after all workers finish")
    args = parser.parse_args()
    run_dashboard(args.state, args.activity, args.interval)


if __name__ == "__main__":
    main()
```

- [ ] **步骤 3：运行测试**

```bash
cd primus-optimizer && python -m pytest tests/test_dashboard.py -v
```
预期结果：全部 4 个测试通过

- [ ] **步骤 4：提交**

```bash
git add primus-optimizer/tools/dashboard.py primus-optimizer/tests/test_dashboard.py
git commit -m "feat(optimizer): rich terminal dashboard with worker table and activity panel"
```

---

## 任务 9：模板

**文件：**
- 创建：`primus-optimizer/templates/round-report.md`
- 创建：`primus-optimizer/templates/pr-description.md`
- 创建：`primus-optimizer/templates/final-summary.md`

- [ ] **步骤 1：创建轮次报告模板**

创建 `primus-optimizer/templates/round-report.md`，使用规格文档中的完整模板（第 4 节，轮次报告模板）。Worker Agent 在编写每轮 report.md 时遵循此模板。

- [ ] **步骤 2：创建 PR 描述模板**

创建 `primus-optimizer/templates/pr-description.md`，使用规格文档中的完整模板（第 11 节，PR 描述模板）。

- [ ] **步骤 3：创建最终总结模板**

创建 `primus-optimizer/templates/final-summary.md`：

```markdown
# Optimization Session Summary — {session_id}

**Hardware**: {hw}
**Duration**: {elapsed}
**Workers**: {num_workers} ({num_completed} completed, {num_failed} failed)

## Results Overview

| Worker | Status | Rounds | Baseline TFLOPS | Final TFLOPS | Geomean Gain | Best Round |
|--------|--------|--------|----------------|-------------|-------------|-----------|
{worker_rows}

## Cross-Pollination Discoveries

{cross_pollination_findings}

## Backend Dispatch Recommendations

{dispatch_recommendations}

## Remaining Optimization Opportunities

{remaining_opportunities}

## Merge Plan

See: `merge-plan.md`

## PR Descriptions

{pr_description_links}
```

- [ ] **步骤 4：提交**

```bash
git add primus-optimizer/templates/
git commit -m "feat(optimizer): report and PR description templates"
```

---

## 任务 10：钩子

**文件：**
- 创建：`primus-optimizer/hooks/pre-benchmark.sh`
- 创建：`primus-optimizer/hooks/post-round.sh`

- [ ] **步骤 1：创建基准测试前钩子**

创建 `primus-optimizer/hooks/pre-benchmark.sh`：

```bash
#!/bin/bash
# 基准测试前环境检查
# 在运行基准测试之前验证 GPU 可用性和环境

set -e

GPU_ID="${HIP_VISIBLE_DEVICES:-0}"

# 检查 GPU 是否可访问
if ! rocm-smi --showid -d "$GPU_ID" &>/dev/null 2>&1; then
    echo "WARNING: GPU $GPU_ID may not be accessible via rocm-smi"
fi

# 检查该 GPU 上是否有其他高负载进程
GPU_UTIL=$(rocm-smi --showuse -d "$GPU_ID" 2>/dev/null | grep -oP '\d+(?=%)' | head -1 || echo "0")
if [ "${GPU_UTIL:-0}" -gt 50 ]; then
    echo "WARNING: GPU $GPU_ID utilization at ${GPU_UTIL}% — benchmark results may be unreliable"
fi

echo "Pre-benchmark check passed for GPU $GPU_ID"
```

- [ ] **步骤 2：创建轮次后钩子**

创建 `primus-optimizer/hooks/post-round.sh`：

```bash
#!/bin/bash
# 轮次后钩子：每轮优化完成后更新状态并记录活动日志
# 用法: post-round.sh <state_json_path> <worker_id> <round_num> <status>

set -e

STATE_PATH="$1"
WORKER_ID="$2"
ROUND_NUM="$3"
STATUS="$4"

if [ -z "$STATE_PATH" ] || [ -z "$WORKER_ID" ]; then
    echo "Usage: post-round.sh <state_path> <worker_id> <round_num> <status>"
    exit 1
fi

echo "Post-round hook: worker=$WORKER_ID round=$ROUND_NUM status=$STATUS"

# 验证轮次输出文件是否存在
ROUND_DIR="$(dirname "$STATE_PATH")/${WORKER_ID}/round-${ROUND_NUM}"
for f in report.md accuracy.log; do
    if [ ! -f "$ROUND_DIR/$f" ]; then
        echo "WARNING: Missing $ROUND_DIR/$f"
    fi
done
```

- [ ] **步骤 3：设置钩子可执行权限并提交**

```bash
chmod +x primus-optimizer/hooks/*.sh
git add primus-optimizer/hooks/
git commit -m "feat(optimizer): pre-benchmark and post-round hooks"
```

---

## 任务 11：Worker 技能

**文件：**
- 创建：`primus-optimizer/skills/optimize-worker/SKILL.md`

- [ ] **步骤 1：编写 Worker SKILL.md**

创建 `primus-optimizer/skills/optimize-worker/SKILL.md`。这是指导 Claude Code 子 Agent 执行优化循环的提示文件，应包含：

1. 角色定义（"You are a Primus-Turbo optimization Worker Agent"）
2. 完整的状态机（INIT -> PROFILE -> ANALYZE -> PLAN -> IMPLEMENT -> VERIFY -> DECIDE -> 循环）
3. 约束条件（每轮一个原子变更、精度门控、最大轮次、瓶颈上报）
4. 输出要求（轮次目录结构、报告模板引用、activity.jsonl 日志记录）
5. GPU 隔离指令（所有基准测试/分析命令使用 `HIP_VISIBLE_DEVICES={gpu_id}`）
6. 基准测试和精度命令（来自 `benchmark_runner.py` 的精确 CLI）
7. 技能知识加载指令（从 `.cursor/skills/` 读取算子特定技能）
8. 终止条件表
9. 报告模板引用，指向 `primus-optimizer/templates/round-report.md`

该技能应编写为完整的提示指令 -- Agent 独立执行所需的一切信息。

- [ ] **步骤 2：提交**

```bash
git add primus-optimizer/skills/optimize-worker/
git commit -m "feat(optimizer): worker agent skill with optimization loop prompt"
```

---

## 任务 12：Review 技能

**文件：**
- 创建：`primus-optimizer/skills/optimize-review/SKILL.md`

- [ ] **步骤 1：编写 Review SKILL.md**

创建 `primus-optimizer/skills/optimize-review/SKILL.md`。该提示指导 Claude Code 子 Agent 执行审查和交叉传播，应包含：

1. 角色定义（"You are the Review Agent"）
2. 输入：读取 `optimizer-state.json` 和所有 Worker 的轮次报告
3. 交叉传播检查清单（Worker 之间的策略可迁移性）
4. 评估标准（策略质量、效率、遗漏的优化机会）
5. 输出：审查决策（continue / switch strategy / terminate / escalate）
6. 最终审查模式：生成 `merge-plan.md`、`final-summary.md` 和 `pr-descriptions/*.md`
7. PR 描述模板引用，指向 `primus-optimizer/templates/pr-description.md`

- [ ] **步骤 2：提交**

```bash
git add primus-optimizer/skills/optimize-review/
git commit -m "feat(optimizer): review agent skill with cross-pollination prompt"
```

---

## 任务 13：知识挖掘技能

**文件：**
- 创建：`primus-optimizer/skills/optimize-knowledge/SKILL.md`

- [ ] **步骤 1：编写 Knowledge SKILL.md**

创建 `primus-optimizer/skills/optimize-knowledge/SKILL.md`。该提示指导 Claude Code 子 Agent 进行优化知识挖掘，应包含：

1. 角色定义（"You are the Knowledge Mining Agent"）
2. 输入：算子、后端、架构、当前瓶颈描述
3. 搜索策略：从 `config/optimizer.yaml` 读取搜索模板
4. Web 搜索指令（使用 WebSearch/WebFetch 工具）
5. GitHub 仓库检查（使用 `gh api` 查看最近的提交）
6. 输出结构：`knowledge_docs/{operator}-{backend}/`，包含 `web_findings.md`、`new_strategies.md`、`references.json`
7. 策略提炼：优先输出可操作的、具体的策略，而非泛泛的建议

- [ ] **步骤 2：提交**

```bash
git add primus-optimizer/skills/optimize-knowledge/
git commit -m "feat(optimizer): knowledge mining skill for bottleneck breakthrough"
```

---

## 任务 14：Coordinator 技能（主入口）

**文件：**
- 创建：`primus-optimizer/skills/optimize/SKILL.md`

- [ ] **步骤 1：编写 Coordinator SKILL.md**

创建 `primus-optimizer/skills/optimize/SKILL.md`。这是 `/optimize` 的主入口技能，应包含：

1. CLI 参数解析指令（--tasks、--operators、--backends、--hw、--profile、--resume、--max-rounds）
2. INIT 阶段：通过 `python -c "from tools.config_loader import ..."` 加载配置，生成会话 ID
3. DISPATCH 阶段：对每个任务，使用 Claude Code 的 `Agent` 工具（设置 `isolation: "worktree"` 和 `run_in_background: true`）启动 Worker Agent，传递 Worker SKILL.md 内容和任务参数
4. MONITOR 阶段：使用 `CronCreate` 或定期 `Bash` 读取 `optimizer-state.json` 来轮询 Worker 状态
5. GPU 队列管理：当 Worker 完成时，释放其 GPU 并启动排队中的任务
6. REVIEW 触发：在里程碑（5 轮）、瓶颈、卡住或全部完成时触发
7. 瓶颈升级：Level 1（指示 Worker 运行性能分析命令）、Level 2（启动 Knowledge Agent）、Level 3（启动 Review Agent 进行最终判定）
8. FINALIZE 阶段：以最终审查模式启动 Review Agent，生成合并计划和 PR 描述
9. 仪表盘启动：为用户打印 `python primus-optimizer/tools/dashboard.py --state ...` 命令

- [ ] **步骤 2：提交**

```bash
git add primus-optimizer/skills/optimize/
git commit -m "feat(optimizer): coordinator skill — main /optimize entry point"
```

---

## 任务 15：集成测试 + Claude Code 配置

**文件：**
- 创建：`primus-optimizer/tests/test_integration.py`
- 修改：`.claude/settings.json`（如存在）或编写手动配置文档

- [ ] **步骤 1：编写集成测试**

创建 `primus-optimizer/tests/test_integration.py`：

```python
"""Integration test: verify all components work together with mock data."""

import json
import os
import tempfile
import pytest
from tools.config_loader import load_config, parse_tasks
from tools.state_store import StateStore, WorkerStatus
from tools.gpu_pool import GPUPool
from tools.bottleneck_detector import BottleneckDetector
from tools.activity_logger import ActivityLogger
from tools.benchmark_runner import BenchmarkRunner


def test_full_lifecycle():
    """Simulate a complete optimization session lifecycle."""
    with tempfile.TemporaryDirectory() as tmpdir:
        state_path = os.path.join(tmpdir, "optimizer-state.json")
        activity_path = os.path.join(tmpdir, "gemm-triton", "activity.jsonl")

        # 1. 解析任务
        tasks = parse_tasks(tasks_str="gemm:triton,gemm:ck")
        assert len(tasks) == 2

        # 2. 初始化 GPU 资源池
        pool = GPUPool([0, 1])

        # 3. 初始化状态
        store = StateStore(state_path)
        store.init_session("test-001", "mi355x", tasks, [0, 1])

        # 4. 分配 GPU 并添加 Worker
        for t in tasks:
            task_id = f"{t['operator']}:{t['backend']}"
            gpu = pool.acquire(task_id)
            store.pop_task_queue()
            store.add_worker(task_id, gpu, f"opt/{task_id.replace(':', '-')}-001")

        state = store.load()
        assert len(state["workers"]) == 2
        assert state["task_queue"] == []

        # 5. 记录轮次
        store.record_round("gemm:triton", 1, "persistent_kernel", 488.1, 614.2, "success")
        store.record_round("gemm:triton", 2, "block_m_tuning", 614.2, 710.5, "success")
        store.record_round("gemm:triton", 3, "xcd_swizzle", 710.5, 720.3, "success")

        # 6. 活动日志
        os.makedirs(os.path.dirname(activity_path), exist_ok=True)
        logger = ActivityLogger(activity_path, "gemm:triton")
        logger.log("VERIFY", 1, "benchmark complete: 614.2 TFLOPS (+25.8%)")

        entries = ActivityLogger.read_recent(activity_path)
        assert len(entries) == 1

        # 7. 瓶颈检测
        detector = BottleneckDetector(threshold=0.02, patience=3)
        state = store.load()
        rounds = state["workers"]["gemm:triton"]["rounds"]
        result = detector.check(rounds)
        assert result.is_bottleneck is False  # 增益仍然显著

        # 8. 模拟低增益轮次触发瓶颈
        store.record_round("gemm:triton", 4, "num_warps", 720.3, 725.1, "success")
        store.record_round("gemm:triton", 5, "async_copy", 725.1, 728.0, "success")
        store.record_round("gemm:triton", 6, "lds_padding", 728.0, 730.2, "success")

        state = store.load()
        rounds = state["workers"]["gemm:triton"]["rounds"]
        result = detector.check(rounds)
        assert result.is_bottleneck is True
        assert result.reason == "diminishing_returns"

        # 9. 更新状态
        store.update_worker_status("gemm:triton", WorkerStatus.BOTTLENECK, "L1")

        # 10. 完成并释放 GPU
        store.update_worker_status("gemm:triton", WorkerStatus.COMPLETED)
        pool.release("gemm:triton")
        assert pool.available_count() == 1

        # 11. 基准测试运行器命令生成
        runner = BenchmarkRunner(repo_root="/shared_nfs/yaoc/agent_work/Primus-Turbo", gpu_id=0)
        cmd = runner.build_benchmark_cmd("gemm", "fp8", "blockwise")
        assert "HIP_VISIBLE_DEVICES=0" in cmd
        assert "bench_gemm_turbo.py" in cmd
```

- [ ] **步骤 2：运行集成测试**

```bash
cd primus-optimizer && python -m pytest tests/test_integration.py -v
```
预期结果：通过

- [ ] **步骤 3：运行完整测试套件**

```bash
cd primus-optimizer && python -m pytest tests/ -v
```
预期结果：全部测试通过（config_loader: 5、state_store: 6、gpu_pool: 7、activity_logger: 2、benchmark_runner: 5、bottleneck_detector: 6、profiler: 3、dashboard: 4、integration: 1 = 共 39 个测试）

- [ ] **步骤 4：编写 Claude Code 配置文档**

在项目 README 或 CLAUDE.md 中添加技能注册说明：

```bash
# 方法 1：将技能符号链接到 Claude Code 目录
ln -s $(pwd)/primus-optimizer/skills/optimize ~/.claude/skills/optimize
ln -s $(pwd)/primus-optimizer/skills/optimize-worker ~/.claude/skills/optimize-worker
ln -s $(pwd)/primus-optimizer/skills/optimize-review ~/.claude/skills/optimize-review
ln -s $(pwd)/primus-optimizer/skills/optimize-knowledge ~/.claude/skills/optimize-knowledge

# 方法 2：添加到 .claude/settings.json
# 技能会从仓库目录结构中自动发现
```

- [ ] **步骤 5：最终提交**

```bash
git add primus-optimizer/tests/test_integration.py
git commit -m "feat(optimizer): integration test + Claude Code setup docs"
```

---

## 验证

完成所有任务后：

1. **单元测试**：`cd primus-optimizer && python -m pytest tests/ -v` -- 全部 39 个测试应通过
2. **仪表盘冒烟测试**：`python primus-optimizer/tools/dashboard.py --state agent_docs/mi355x/optimizer-state.json` -- 应正常启动（或显示"等待状态文件"）
3. **配置加载**：`python -c "from primus_optimizer.tools.config_loader import load_config; print(load_config())"` -- 应打印默认配置
4. **技能文件检查**：`ls primus-optimizer/skills/*/SKILL.md` -- 应列出 4 个文件
5. **手动端到端测试**：在 Claude Code 中运行 `/optimize --tasks gemm:triton --hw mi355x` -- Coordinator 应解析参数、创建状态文件、在 worktree 中启动 Worker Agent，并打印仪表盘附加命令
