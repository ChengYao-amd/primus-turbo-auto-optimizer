# Primus Optimizer — 设计规范

> 用于在 Primus-Turbo 中实现自动化多智能体算子优化的 Claude Code 插件。

**日期**: 2026-04-07
**状态**: 草案
**方案**: Claude Code 原生多智能体（MVP），逐步演进为 Python 编排器 + Claude Code 工作节点

---

## 目录

1. [概述](#1-概述)
2. [架构](#2-架构)
3. [任务配置](#3-任务配置)
4. [工作智能体](#4-工作智能体)
5. [协调器](#5-协调器)
6. [评审智能体](#6-评审智能体)
7. [瓶颈升级](#7-瓶颈升级)
8. [知识智能体](#8-知识智能体)
9. [终端仪表盘](#9-终端仪表盘)
10. [状态存储](#10-状态存储)
11. [最终输出](#11-最终输出)
12. [用户指南](#12-用户指南)

---

## 1. 概述

### 问题

Primus-Turbo 包含 6 个算子族（GEMM、Attention、Grouped GEMM、MoE、Quantization、Normalization），跨 3 个后端（CK、HipBLASLt、Triton），面向 2 个硬件平台（MI300X、MI355X）。手动逐一优化每种组合效率低下且难以扩展。

### 解决方案

一个 Claude Code 技能插件（`primus-optimizer`），可编排多个并行优化智能体——每个智能体负责一个算子 x 后端的组合——实现自动化性能分析、瓶颈检测、知识挖掘和交叉评审。

### 相关先行工作

- **内部**: `.cursor/` 规则和技能已实现单智能体串行优化循环（Profile -> Analyze -> Plan -> Implement -> Verify -> Decide -> Log），并完成了 11 轮成功优化。
- **外部**: NVIDIA 的 AVO（arXiv:2603.24517）展示了在 Blackwell GPU 上进行 7 天自主智能体驱动的内核演进，相比 cuDNN/FA4 实现了 3.5-10.5% 的性能提升。

### 设计原则

- **原子性**: 每轮仅一个优化点，便于明确归因和回滚。
- **可验证性**: 每项结论均有可复现的命令和数据支撑。
- **可追溯性**: 完整的优化历史记录存储在结构化文件中。
- **并行性**: 算子 x 后端级别的并发，GPU 隔离。
- **渐进性**: 终端仪表盘 MVP，数据格式为未来 Web UI 预留。

---

## 2. 架构

### 组件布局

```
primus-optimizer/
├── skills/
│   ├── optimize/                    # 主入口技能 (/optimize)
│   │   └── SKILL.md
│   ├── optimize-worker/             # 工作智能体技能
│   │   └── SKILL.md
│   ├── optimize-review/             # 评审智能体技能
│   │   └── SKILL.md
│   └── optimize-knowledge/          # 知识挖掘技能
│       └── SKILL.md
│
├── tools/
│   ├── gpu_pool.py                  # GPU 资源池管理
│   ├── benchmark_runner.py          # 标准化基准测试执行器
│   ├── profiler.py                  # 细粒度性能分析封装
│   ├── bottleneck_detector.py       # 瓶颈检测（多维度）
│   ├── state_store.py               # 优化状态持久化（JSON）
│   └── dashboard.py                 # 终端仪表盘（rich）
│
├── hooks/
│   ├── pre-benchmark.sh             # 基准测试前的环境检查
│   └── post-round.sh               # 每轮结束后的状态更新
│
├── templates/
│   ├── round-report.md              # 每轮报告模板
│   ├── pr-description.md            # PR 描述模板
│   └── final-summary.md             # 最终总结模板
│
└── config/
    └── optimizer.yaml               # 全局配置（目标、GPU 映射、配置文件）
```

### 数据流

```
User: /optimize --tasks gemm:triton,attn:ck --hw mi355x

                    ┌─────────────────────────────────────────────┐
                    │            Coordinator (optimize skill)      │
                    │  1. 解析配置 -> 任务矩阵                      │
                    │  2. 检查 GPU 池 -> 分配 GPU                   │
                    │  3. 派发工作智能体（后台运行，                   │
                    │     worktree 隔离）                           │
                    │  4. 监控循环 -> 更新仪表盘                      │
                    │  5. 遇到瓶颈/完成 -> 评审智能体                  │
                    │  6. 收尾 -> 合并计划 + PR 描述                  │
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
        Profile -> Analyze -> Implement -> Verify -> Log (循环)
```

---

## 3. 任务配置

### 命令行接口

可通过三种方式指定优化目标：

**显式任务列表（推荐）**：
```bash
/optimize --tasks gemm:ck,attention:triton --hw mi355x
```

**笛卡尔积展开**：
```bash
/optimize --operators gemm,attention --backends triton,ck --hw mi355x
# 展开为: gemm:triton, gemm:ck, attention:triton, attention:ck
```

**混合模式**：
```bash
/optimize --tasks gemm:ck,gemm:hipblaslt --operators attention --backends triton --hw mi355x
# 结果: gemm:ck, gemm:hipblaslt, attention:triton
```

**命名配置文件**：
```bash
/optimize --profile mi355x-priority
```

**恢复中断的会话**：
```bash
/optimize --resume                          # 恢复上一次会话
/optimize --resume --session 2026-04-07-001 # 恢复指定会话
```
协调器读取 `optimizer-state.json`，检测每个工作智能体最后提交的轮次，并从该处重新启动。

### 配置文件（`config/optimizer.yaml`）

```yaml
# GPU 资源配置
gpu_pool:
  device_type: hip              # hip 或 cuda
  device_ids: [0, 1, 2, 3]     # 可用 GPU ID

# 全局默认值
defaults:
  max_rounds: 10                # 每个工作智能体的最大优化轮数
  bottleneck_threshold: 0.02    # 2% — 连续增益低于此值触发瓶颈
  bottleneck_patience: 3        # 连续低增益轮数达到此值后触发瓶颈
  accuracy_snr_bf16: 30         # BF16 的 dB 阈值
  accuracy_snr_fp8: 20          # FP8 的 dB 阈值
  monitor_interval: 60          # 仪表盘刷新间隔（秒）

# 命名配置文件
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

# 知识智能体搜索配置
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

---

## 4. 工作智能体

### 优化循环状态机

```
                    ┌──────────┐
                    │  INIT    │ 加载技能知识库 + 建立基线
                    └────┬─────┘
                         ▼
              ┌─────────────────────┐
              │     PROFILE         │ 运行基准测试 -> 收集性能数据
              └─────────┬───────────┘
                        ▼
              ┌─────────────────────┐
              │     ANALYZE         │ 识别瓶颈类型（计算/内存/延迟）
              │                     │ 与 roofline / SOTA 对比
              └─────────┬───────────┘
                        ▼
              ┌─────────────────────┐
              │     PLAN            │ 从技能知识库中选择策略
              │                     │ 设计单一原子修改
              └─────────┬───────────┘
                        ▼
              ┌─────────────────────┐
              │     IMPLEMENT       │ 修改代码（单一原子变更）
              │                     │ 在 worktree 分支上 git commit
              └─────────┬───────────┘
                        ▼
              ┌─────────────────────┐
              │     VERIFY          │ 第 1 级：快速精度检查
              │                     │ 第 2 级：完整基准测试
              │                     │ 第 3 级：回归检测
              └─────────┬───────────┘
                   ┌────┴────┐
                   ▼         ▼
              [通过]      [失败] --> git revert -> 返回 PLAN（切换策略）
                   │                  连续 3 次失败 -> 上报协调器
                   ▼
              ┌─────────────────────┐
              │     DECIDE          │ 记录增益，决定是否继续
              └─────────┬───────────┘
                   ┌────┴──────────┐
                   ▼               ▼
           [继续]            [瓶颈]
           返回 PROFILE      通知协调器
```

### 技能加载

每个工作智能体在启动时加载特定算子的知识：

```
Worker(gemm:triton) 加载:
  ├── skills/triton-gemm-optimize/SKILL.md      # 优化指南
  ├── skills/amd-gpu-architecture/SKILL.md       # 硬件约束
  ├── skills/kernel-profiling/SKILL.md           # 性能分析方法论
  └── skills/sota-knowledge-base/SKILL.md        # SOTA 参考资料

Worker(attention:triton) 加载:
  ├── skills/triton-attention-optimize/SKILL.md
  ├── skills/amd-gpu-architecture/SKILL.md
  └── ...
```

### 工作智能体提示结构

```markdown
## 你的任务
你是一个 Primus-Turbo 优化循环工作智能体。
- 目标算子: {operator}（{backend} 后端）
- 目标硬件: {hw}
- 分配的 GPU: HIP_VISIBLE_DEVICES={gpu_id}
- 工作分支: {worktree_branch}

## 约束条件
- 每轮仅一个原子修改
- 修改前后均须通过精度检查
- 最大轮数: {max_rounds}
- 连续 {bottleneck_patience} 轮增益低于 {bottleneck_threshold} 后上报瓶颈

## 当前状态
- 已完成轮数: {completed_rounds}
- 当前基线: {current_tflops} TFLOPS
- 已尝试策略: {tried_strategies}

## 知识库
{loaded_skill_content}

## 输出要求
每轮结束后，写入以下目录：
  agent_docs/{hw}/{operator}-{backend}/round-{N}/
    ├── report.md        # 轮次报告（见下方模板）
    ├── baseline.json    # 修改前性能
    ├── optimized.json   # 修改后性能
    └── accuracy.log     # 精度验证日志
```

### GPU 隔离

每个工作智能体通过环境变量获得独占的 GPU：

```python
# tools/gpu_pool.py
class GPUPool:
    def __init__(self, gpu_ids: list[int]):
        self.available = set(gpu_ids)
        self.allocated = {}  # task_id -> gpu_id

    def acquire(self, task_id: str) -> int:
        gpu_id = self.available.pop()
        self.allocated[task_id] = gpu_id
        return gpu_id

    def release(self, task_id: str):
        gpu_id = self.allocated.pop(task_id)
        self.available.add(gpu_id)
```

工作智能体基准测试执行：
```bash
HIP_VISIBLE_DEVICES={gpu_id} python benchmark/ops/bench_gemm_turbo.py ...
```

### 终止条件

| 条件 | 行为 |
|------|------|
| 达到 `max_rounds` | 正常完成，报告最终结果 |
| 连续 3 轮增益 <2% | 上报 `bottleneck`，等待协调器决策 |
| 连续 3 次 VERIFY 失败 | 上报 `stuck`，协调器介入 |
| Roofline 利用率 >80% | 上报 `near_optimal`，建议终止 |
| 与 SOTA 差距 <5% | 上报 `competitive`，建议终止 |

### 轮次报告模板（`round-{N}/report.md`）

```markdown
# 第 {N} 轮: {优化策略名称}

## 优化理由

### 问题分析
{根据性能分析数据识别出的具体瓶颈，例如：
"当前持久化内核使用 BLOCK_M=128，在 M<512 的解码场景中每个 CU 仅分配 1-2 个 tile，
导致 304 个 CU 中 >60% 处于空闲状态，
计算利用率仅为 9.8%"}

### 优化策略
{策略的技术原理，例如：
"将 BLOCK_M 从 128 降低至 64，使 tile 数量翻倍。对于 M=256, N=7168，
tile 数从 2x56=112 增加至 4x56=224，CU 利用率从 36.8% 提升至 73.7%。
代价是单个 tile 的计算密度降低，但对于小 M 场景，CU 利用率的提升
远超这一代价。"}

### 理论预期
{预期提升及依据，例如："基于 CU 利用率从 36.8% 到 73.7% 的线性
外推，预期提升 +30-50%"}

## 代码变更

### 修改文件
- `primus_turbo/triton/gemm/gemm_kernel.py:L142-L158`

### Diff 摘要
{关键 diff，非完整列表}

### 提交记录
- 分支: `opt/{operator}-{backend}-{session_id}`
- 提交: `{hash}` - `{message}`

## 复现命令

### 精度验证
```bash
# 生成 accuracy.log
HIP_VISIBLE_DEVICES={gpu_id} python benchmark/accuracy/eval_gemm_accuracy.py \
    --dtype {dtype} --quant {quant} \
    --shapes "{shape_list}" \
    --backend {backend} \
    2>&1 | tee agent_docs/{hw}/{operator}-{backend}/round-{N}/accuracy.log
```

### 基线性能
```bash
# 生成 baseline.json（从修改前的状态运行）
git stash  # 或 git checkout HEAD~1
HIP_VISIBLE_DEVICES={gpu_id} python benchmark/ops/bench_{operator}_turbo.py \
    --dtype {dtype} --quant {quant} \
    --shapes "{shape_list}" \
    --backend {backend} --warmup 10 --rep 50 \
    --output agent_docs/{hw}/{operator}-{backend}/round-{N}/baseline.json
git stash pop  # 或 git checkout {worktree_branch}
```

### 优化后性能
```bash
# 生成 optimized.json（从修改后的状态运行）
HIP_VISIBLE_DEVICES={gpu_id} python benchmark/ops/bench_{operator}_turbo.py \
    --dtype {dtype} --quant {quant} \
    --shapes "{shape_list}" \
    --backend {backend} --warmup 10 --rep 50 \
    --output agent_docs/{hw}/{operator}-{backend}/round-{N}/optimized.json
```

## 结果

| Shape (M,N,K) | 基线 (TFLOPS) | 优化后 (TFLOPS) | 变化 |
|---|---|---|---|
| 256,7168,7168 | 488.1 | 793.2 | +62.5% |
| ... | ... | ... | ... |

- **几何平均提升**: +X.X%
- **峰值提升**: +X.X% @ shape (M,N,K)
- **回归检测**: 无回归 / {shape} 回归了 X%

## 结论
{成功/失败，是否采纳，下一步方向}
```

---

## 5. 协调器

### 生命周期

```
INIT --> DISPATCH --> MONITOR --> REVIEW --> FINALIZE

INIT:     加载 optimizer.yaml / 命令行参数
DISPATCH: 解析配置 -> 任务矩阵，分配 GPU，启动工作智能体
MONITOR:  轮询工作智能体状态，更新仪表盘，管理 GPU 队列
REVIEW:   在里程碑 / 瓶颈 / 完成时触发评审智能体
FINALIZE: 合并结果，生成 PR 描述，最终总结
```

### 监控循环

每 `monitor_interval` 秒（默认 60 秒）：

```
1. 读取 optimizer-state.json
2. 检查每个工作智能体的状态：
   - running     -> 更新仪表盘显示
   - bottleneck  -> 触发瓶颈升级流程
   - stuck       -> 触发评审智能体介入
   - completed   -> 回收 GPU，检查是否有排队任务
3. 若工作智能体完成且有排队任务 -> 分配释放的 GPU，启动新工作智能体
4. 所有工作智能体完成 -> 进入 FINALIZE
```

### GPU 队列管理

当任务数量超过 GPU 数量时，协调器维护一个 FIFO 队列：

```
GPU 池: [0, 1, 2, 3]
任务队列: [gemm:triton, gemm:ck, attn:triton, attn:ck, grouped_gemm:triton]

阶段 1: 前 4 个任务各占 1 个 GPU，并行运行
阶段 2: gemm:triton 最先完成 -> GPU:0 释放 -> grouped_gemm:triton 启动
```

---

## 6. 评审智能体

### 触发条件

| 触发条件 | 评审范围 |
|----------|----------|
| 任一工作智能体完成 5 轮 | 里程碑评审：策略方向、遗漏的机会 |
| 工作智能体上报 `bottleneck` | 瓶颈诊断 + 决策（见第 7 节） |
| 工作智能体上报 `stuck` | 失败分析，建议新策略或终止 |
| 所有工作智能体完成 | 最终总结：交叉借鉴 + 合并报告 |

### 评审智能体提示结构

```markdown
## 你的角色
你是 Primus-Turbo 优化循环评审智能体。你的职责包括：
1. 审查工作智能体的优化历史，评判策略质量
2. 识别交叉借鉴机会（工作智能体 A 的成功策略
   可能可以迁移到工作智能体 B）
3. 在瓶颈诊断时做出行动决策

## 所有工作智能体状态
{从 optimizer-state.json 加载}

## 待评审的工作智能体
{触发本次评审的工作智能体的完整轮次历史}

## 交叉借鉴检查清单
- 工作智能体 A 成功使用了策略 X -> 工作智能体 B 是否也可以应用？
  （例如，gemm:triton 的持久化内核 -> 是否适用于 grouped_gemm:triton？）
- 相同算子在不同后端之间的性能差异 -> 这是否表明某些后端
  更适合特定的 shape 范围？
  （例如，gemm:ck 在 M<512 时表现更优 -> 是否应更新后端分发规则？）

## 输出要求
1. 对每个工作智能体的评估（策略方向、效率、遗漏的机会）
2. 交叉借鉴建议（如有）
3. 决策建议：继续 / 切换策略 / 终止
```

---

## 7. 瓶颈升级

当工作智能体上报 `bottleneck`（连续 3 轮增益 <2%）时，系统触发三级升级：

### 第 1 级：细粒度性能分析

协调器指示工作智能体运行深度性能分析：

```
1. rocprof --stats: 内核级时间统计
2. omniperf: 硬件计数器采集
   - 计算利用率
   - 内存带宽利用率
   - LDS bank 冲突
   - MFMA 流水线利用率
3. 生成 Roofline 分析
```

性能分析结果注入工作智能体上下文。工作智能体携带新数据重新进入 ANALYZE -> PLAN。

### 第 2 级：知识挖掘

若第 1 级未能突破瓶颈（再次触发瓶颈），协调器启动知识智能体：

```
1. 网络搜索最新优化技术：
   - "{operator} optimization {hw_arch} 2025 2026"
   - "triton kernel optimization AMD CDNA"
   - GitHub: 相关仓库的最新提交
2. 获取并分析：
   - AITER / vLLM / FlashAttention 最新更新日志
   - AMD ROCm 文档更新
   - 相关论文（arXiv）
3. 提炼可操作的策略列表，注入工作智能体上下文
```

知识智能体输出：
```
knowledge_docs/{operator}-{backend}/
  ├── web_findings.md     # 搜索发现
  ├── new_strategies.md   # 提炼的新策略
  └── references.json     # 来源 URL 列表
```

### 第 3 级：评审智能体最终裁定

若第 2 级仍未能突破瓶颈（第三次瓶颈），评审智能体进行综合判断：

1. 内核是否已接近硬件理论极限？（roofline >80%）
2. 与 SOTA 的差距是否 <5%？
3. 是否存在跨工作智能体的交叉借鉴机会？

决策：
- **TERMINATE**: 当前算子 x 后端已接近极限，停止。
- **PIVOT**: 切换到根本不同的策略方向。
- **ESCALATE**: 标记为需要人工介入。

---

## 8. 知识智能体

### 搜索策略配置

在 `config/optimizer.yaml` 的 `knowledge` 部分定义。每个算子的搜索模板包括：

- **关键词查询**用于网络搜索（算子 + 后端 + 架构相关）
- **待检查的 GitHub 仓库**，查看最近的提交和发布
- **arXiv 搜索**相关论文

### 输出结构

```markdown
# 知识挖掘报告: {operator}:{backend}

## 执行的搜索查询
1. "{query_1}" -> {N} 个结果
2. "{query_2}" -> {N} 个结果

## 关键发现

### 发现 1: {标题}
- **来源**: {URL}
- **相关性**: {为什么这对我们的优化有意义}
- **可应用的策略**: {从此发现中提炼的具体优化思路}

### 发现 2: ...

## 推荐的新策略（按优先级排序）
1. {策略}: {描述} — 预期影响: {估计}
2. ...

## 参考资料
- [{title}]({url}) — {一行相关性说明}
```

---

## 9. 终端仪表盘

### 启动机制

仪表盘是一个**独立的只读进程**，与协调器完全解耦。它读取 `optimizer-state.json` 和 `activity.jsonl` 文件——无需 IPC 或共享内存。

**自动启动**：协调器启动时，会在后台生成仪表盘进程并打印连接命令：

```
Coordinator: Dashboard started. Attach in another terminal:
  python primus-optimizer/tools/dashboard.py --state agent_docs/mi355x/optimizer-state.json
```

**手动启动**：用户可随时独立启动：

```bash
# 在另一个终端中（同时 Claude Code 中正在运行 /optimize）
python primus-optimizer/tools/dashboard.py \
    --state agent_docs/mi355x/optimizer-state.json \
    --activity "agent_docs/mi355x/*/activity.jsonl"
```

**仪表盘生命周期**：
- 仪表盘默认以监视模式启动，轮询文件以获取更新
- 若 `optimizer-state.json` 尚不存在，仪表盘将等待其出现
- 当所有工作智能体达到终态（`completed`/`failed`）时，仪表盘显示最终总结并退出（或使用 `--keep-alive` 保持运行）
- 仪表盘崩溃或关闭对优化过程**零影响**——协调器和工作智能体独立继续运行

### 布局

```
┌─ Primus Optimizer ──────────────────────────────────────────────────────────┐
│ 会话: 2026-04-07-001 │ 硬件: MI355X │ 已用时间: 02:34:17 │ GPU: 4/4 忙碌  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  工作智能体状态                                                               │
│  ┌──────────────────┬────────┬─────┬──────────┬───────────┬────────────┐   │
│  │ 任务             │ 状态   │ GPU │ 轮次     │ TFLOPS    │ 增益       │   │
│  ├──────────────────┼────────┼─────┼──────────┼───────────┼────────────┤   │
│  │ gemm:triton      │ ▶ 运行 │  0  │ 5/10     │ 488→793   │ +62.5%     │   │
│  │ gemm:ck          │ ▶ 运行 │  1  │ 3/10     │ 429→512   │ +19.3%     │   │
│  │ attn:triton      │ ⚠ 瓶颈 │  2  │ 7/8  L1  │ 301→452   │ +50.2%     │   │
│  │ grouped_gemm:ck  │ ◻ 等待 │  -  │ -        │ -         │ -          │   │
│  └──────────────────┴────────┴─────┴──────────┴───────────┴────────────┘   │
│                                                                             │
│  实时活动                                                                    │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ 12:34:17 [gemm:triton  ] 第 5 轮 VERIFY 通过 (+8.3%)               │  │
│  │ 12:33:42 [attn:triton  ] 瓶颈 L1: 正在运行 omniperf...              │  │
│  │ 12:32:08 [gemm:ck      ] 第 3 轮 IMPLEMENT: M 感知 tile 选择       │  │
│  │ 12:31:55 [gemm:triton  ] 第 5 轮 IMPLEMENT: BLOCK_K=128->256       │  │
│  │ 12:30:01 [attn:triton  ] 第 7 轮增益 +0.8% < 2%，触发瓶颈          │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  性能趋势 (gemm:triton)                                                     │
│  TFLOPS                                                                     │
│  800 ┤                                                          ╭──● R5    │
│  700 ┤                                              ╭───────────╯          │
│  600 ┤                              ╭───────────────╯                      │
│  500 ┤              ╭───────────────╯                                      │
│  488 ┤──────●───────╯                                                      │
│      └──────┬───────┬───────────────┬───────────────┬───────────┬───       │
│            R0      R1              R2              R3           R4          │
│                                                                             │
│  [q] 退出  [r] 刷新  [w] 切换工作智能体  [p] 暂停工作智能体  [d] 详情     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 组件

| 组件 | 数据来源 | 刷新频率 |
|------|----------|----------|
| 页眉 | `optimizer-state.json` 会话信息 | 每次刷新 |
| 工作智能体状态表 | `optimizer-state.json` workers 字段 | 60 秒 |
| 实时活动 | 工作智能体 `activity.jsonl` 文件（聚合） | 10 秒 |
| 性能趋势 | 轮次 `optimized.json` 文件 | 轮次完成时 |

### 状态图标

```
▶ RUN   — 正常运行中
⚠ BTNK  — 瓶颈处理中（标注 L1/L2/L3）
✓ DONE  — 优化完成
✗ FAIL  — 失败 / 需要人工介入
◻ WAIT  — 排队中，等待 GPU
⏸ PAUSE — 已被用户暂停
```

### 活动日志格式

每个工作智能体追加写入自己的 `activity.jsonl`：

```jsonl
{"ts":"2026-04-07T12:34:17Z","worker":"gemm:triton","phase":"VERIFY","round":5,"msg":"accuracy check passed, 3344/3344 tests OK"}
{"ts":"2026-04-07T12:34:18Z","worker":"gemm:triton","phase":"VERIFY","round":5,"msg":"benchmark complete: 793.2 TFLOPS (+8.3%)"}
{"ts":"2026-04-07T12:34:19Z","worker":"gemm:triton","phase":"DECIDE","round":5,"msg":"improvement +8.3% > 2%, continuing"}
```

### 键盘控制

| 按键 | 操作 |
|------|------|
| `q` | 优雅退出（等待当前轮次完成，然后停止所有工作智能体） |
| `Ctrl+C` | 立即中断（工作智能体 worktree 保留，会话可恢复） |
| `w` | 切换性能趋势图表以显示不同的工作智能体 |
| `d` | 展开所选工作智能体的详细轮次历史 |
| `p` | 暂停 / 恢复所选工作智能体 |

### 未来 Web UI 升级路径

所有数据通过 JSON 文件流转——仪表盘是只读的。后续添加 Web UI 只需一层薄的 API 层：

- `optimizer-state.json` -> REST `GET /api/state`
- `activity.jsonl` -> WebSocket 推送
- 轮次 JSON 文件 -> `GET /api/workers/{id}/rounds/{n}`

无需对工作智能体或协调器逻辑进行任何改动。

---

## 10. 状态存储

### `optimizer-state.json`

中心状态文件位于 `agent_docs/{hw}/optimizer-state.json`：

```json
{
  "session_id": "2026-04-07-001",
  "config": {
    "hw": "mi355x",
    "tasks": [
      {"operator": "gemm", "backend": "triton", "max_rounds": 10},
      {"operator": "gemm", "backend": "ck", "max_rounds": 10},
      {"operator": "attention", "backend": "triton", "max_rounds": 8}
    ]
  },
  "gpu_pool": {
    "total": [0, 1, 2, 3],
    "allocated": {"gemm:triton": 0, "gemm:ck": 1, "attn:triton": 2}
  },
  "workers": {
    "gemm:triton": {
      "status": "running",
      "gpu_id": 0,
      "worktree_branch": "opt/gemm-triton-001",
      "current_round": 5,
      "rounds": [
        {
          "round": 1,
          "strategy": "persistent_kernel",
          "baseline_tflops": 488.1,
          "result_tflops": 614.2,
          "improvement_pct": 25.8,
          "status": "success"
        }
      ],
      "bottleneck": null,
      "started_at": "2026-04-07T10:00:00Z"
    }
  },
  "task_queue": ["grouped_gemm:ck"],
  "review_log": [
    {
      "trigger": "milestone",
      "worker": "gemm:triton",
      "round": 5,
      "decision": "continue",
      "cross_pollination": [],
      "timestamp": "2026-04-07T12:00:00Z"
    }
  ],
  "started_at": "2026-04-07T10:00:00Z",
  "updated_at": "2026-04-07T12:30:00Z"
}
```

### 轮次输出目录

```
agent_docs/{hw}/{operator}-{backend}/
├── round-0/                   # 基线轮次
│   ├── report.md
│   ├── baseline.json
│   └── accuracy.log
├── round-1/
│   ├── report.md
│   ├── baseline.json          # 修改前（= round-0 优化后的结果）
│   ├── optimized.json         # 修改后
│   └── accuracy.log
├── round-N/
│   └── ...
└── activity.jsonl             # 工作智能体活动日志（追加写入）
```

---

## 11. 最终输出

### 目录结构

```
agent_docs/{hw}/session-{id}/
├── final-summary.md           # 全局总结报告
├── optimizer-state.json       # 完整状态快照
├── merge-plan.md              # Worktree 合并计划
└── pr-descriptions/           # 每个推荐合并的 PR 描述
    ├── gemm-triton.md
    ├── gemm-ck.md
    └── attention-triton.md
```

### PR 描述模板（`pr-descriptions/{operator}-{backend}.md`）

```markdown
## 摘要

通过 {N} 轮迭代优化，优化 {hw} 上的 {operator}（{backend} 后端），
实现了 **+{geomean}% 几何平均提升**
（{baseline_tflops} -> {optimized_tflops} TFLOPS）。

### 关键变更

- **第 1 轮 -- {strategy_1}**: {一句话描述} (+{pct_1}%)
- **第 3 轮 -- {strategy_3}**: {一句话描述} (+{pct_3}%)
- ...

### 优化原理

{2-3 段技术说明，涵盖：
  - 发现了什么瓶颈（如小 M 场景下 CU 利用率不足）
  - 采取了什么方法及原因（如通过降低 tile 尺寸，以牺牲
    单 tile 计算密度换取更高的 CU 占用率）
  - 为什么这是安全的（精度验证结果）}

## 性能结果

| Shape (M,N,K) | 基线 (TFLOPS) | 优化后 (TFLOPS) | 变化 |
|---|---|---|---|
| 256,7168,7168 | 488.1 | 793.2 | +62.5% |
| ... | ... | ... | ... |

**几何平均**: +{X.X}% | **峰值**: +{X.X}% @ {shape} | **回归**: 无

## 精度验证

- {dtype} 精度：全部 {N} 个测试用例通过（SNR > {threshold} dB）
- 复现：`{accuracy_command}`

## 复现基准测试

```bash
# 基线
{baseline_command}

# 优化后
{optimized_command}
```

## 测试计划

- [ ] 精度检查：`{accuracy_command}`
- [ ] 性能基准测试：`{benchmark_command}`
- [ ] 回归检查：`python tools/check_regression.py --baseline {path}`
```

### 合并计划（`merge-plan.md`）

```markdown
# 合并计划 — 会话 {session_id}

## 推荐合并

| 分支 | 算子:后端 | 几何平均增益 | 回归 | PR 描述 |
|------|----------|-------------|------|---------|
| opt/gemm-triton-001 | gemm:triton | +62.5% | 无 | pr-descriptions/gemm-triton.md |
| opt/gemm-ck-001 | gemm:ck | +19.3% | 无 | pr-descriptions/gemm-ck.md |

## 合并顺序
1. `opt/gemm-triton-001` — 预计无冲突
2. `opt/gemm-ck-001` — 可能在 `gemm_impl.py` 中与 gemm-triton 冲突，需手动审查

## 不推荐合并的分支
| 分支 | 原因 |
|------|------|
| opt/attn-triton-001 | 在 seq_len=8192 时存在回归 (-3.2%)，需要调查 |

## 清理
合并后，移除 worktree：
```bash
git worktree remove .claude/worktrees/opt-gemm-triton-001
git worktree remove .claude/worktrees/opt-gemm-ck-001
```
```

---

## 12. 用户指南

### 前置条件

- 已安装并认证 Claude Code CLI
- 已克隆 Primus-Turbo 仓库且具有 GPU 访问权限
- 已安装 `rich` 的 Python 环境（`pip install rich`）
- 具备 ROCm 和 HIP 工具包的 AMD GPU
- 可用的 `rocprof` 和 `omniperf`，用于性能分析（可选，用于瓶颈 L1）

### 安装

```bash
# 从 Primus-Turbo 仓库根目录
cd primus-optimizer

# 将插件安装为 Claude Code 技能
# （将技能复制到 Claude Code 技能目录或在 settings.json 中配置）
cp -r skills/* ~/.claude/skills/

# 安装工具的 Python 依赖
pip install rich pyyaml

# 验证 GPU 可用性
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

### 快速入门

**1. 运行单个算子优化：**
```bash
# 在 Claude Code 中，优化 MI355X 上使用 Triton 后端的 GEMM
/optimize --tasks gemm:triton --hw mi355x
```

**2. 在多个目标上并行优化：**
```bash
# 并行优化 GEMM (CK) 和 Attention (Triton)
/optimize --tasks gemm:ck,attention:triton --hw mi355x
```

**3. 使用预定义配置文件：**
```bash
# 运行 MI355X 优先优化配置
/optimize --profile mi355x-priority
```

**4. 全面扫描并自定义轮次上限：**
```bash
# 所有算子 x 所有后端，每个最多 5 轮
/optimize --operators gemm,attention,grouped_gemm \
          --backends triton,ck \
          --hw mi300x \
          --max-rounds 5
```

### 配置

编辑 `config/optimizer.yaml` 以自定义：

- **GPU 池**: 可用于优化的 GPU ID
- **配置文件**: 常见场景的预定义优化目标集
- **默认值**: 轮次上限、瓶颈阈值、精度标准
- **知识搜索**: 每个算子的网络搜索模板（用于瓶颈 L2）

### 监控

当 `/optimize` 启动时，协调器会打印仪表盘连接命令。打开一个**单独的终端**并运行：

```bash
python primus-optimizer/tools/dashboard.py --state agent_docs/mi355x/optimizer-state.json
```

仪表盘是只读的且完全独立——关闭它不会影响优化过程。主要交互方式：

| 按键 | 操作 |
|------|------|
| `q` | 优雅关闭——完成当前轮次，然后停止 |
| `Ctrl+C` | 立即停止——worktree 保留以便后续恢复 |
| `w` | 在不同工作智能体之间切换性能图表 |
| `d` | 显示所选工作智能体的详细轮次历史 |
| `p` | 暂停或恢复工作智能体 |

### 查看结果

在优化过程中和完成后，结果存储在结构化目录中：

```bash
# 查看当前会话状态
cat agent_docs/mi355x/optimizer-state.json | python -m json.tool

# 阅读特定轮次的报告
cat agent_docs/mi355x/gemm-triton/round-3/report.md

# 对比基线与优化后的性能
diff <(cat agent_docs/mi355x/gemm-triton/round-3/baseline.json | python -m json.tool) \
     <(cat agent_docs/mi355x/gemm-triton/round-3/optimized.json | python -m json.tool)

# 查看所有工作智能体的实时活动
tail -f agent_docs/mi355x/*/activity.jsonl
```

### 恢复会话

如果会话被中断（Ctrl+C 或崩溃），worktree 和状态会被保留：

```bash
# 恢复上一次会话——协调器读取 optimizer-state.json
# 并从每个工作智能体最后完成的轮次重新启动
/optimize --resume

# 恢复指定会话
/optimize --resume --session 2026-04-07-001
```

协调器会检测每个工作智能体最后提交的轮次，并从该处恢复。

### 合并结果

优化完成后，查看合并计划：

```bash
# 合并计划和 PR 描述位于：
cat agent_docs/mi355x/session-2026-04-07-001/merge-plan.md
cat agent_docs/mi355x/session-2026-04-07-001/pr-descriptions/gemm-triton.md

# 合并推荐的分支：
git merge opt/gemm-triton-001

# 直接创建 PR（使用生成的描述）：
gh pr create --title "Optimize GEMM Triton on MI355X (+62.5%)" \
  --body-file agent_docs/mi355x/session-2026-04-07-001/pr-descriptions/gemm-triton.md
```

### 故障排除

| 问题 | 解决方案 |
|------|----------|
| "No GPUs available" | 检查 `HIP_VISIBLE_DEVICES` 和 `config/optimizer.yaml` 中的 gpu_pool |
| 工作智能体卡在 VERIFY | 检查 `accuracy.log` 是否有失败记录；实验性变更可能需要放宽 SNR 阈值 |
| 仪表盘不更新 | 确认 `optimizer-state.json` 正在被写入；检查工作智能体进程是否存活 |
| 瓶颈 L1 失败（无 omniperf） | 安装 omniperf 或通过 `--skip-profiling` 跳至 L2（知识挖掘） |
| 合并时 worktree 冲突 | 按 `merge-plan.md` 的合并顺序操作；在 `*_impl.py` 分发文件中解决冲突 |
| 会话恢复失败 | 删除过期锁文件：`rm agent_docs/{hw}/optimizer-state.lock` |

### 最佳实践

1. **从小处开始**: 先运行单个 `--tasks gemm:triton` 验证循环流程，再扩展到并行。
2. **使用配置文件**: 在 `optimizer.yaml` 中定义常用的目标集，避免每次输入冗长的命令行参数。
3. **合并前务必审查**: 始终阅读 `merge-plan.md`——并非所有分支都推荐合并。
4. **关注 GPU 显存**: 大规模 Triton autotune 可能耗尽 GPU 显存。设置 `TRITON_AUTOTUNE_MAX_CONFIGS` 以限制编译数量。
5. **保存成功策略**: 成功完成一轮会话后，将新发现的优化技术更新到技能知识库（`skills/*/SKILL.md`），为后续运行提供参考。
