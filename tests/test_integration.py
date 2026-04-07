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

        # 1. Parse tasks
        tasks = parse_tasks(tasks_str="gemm:triton,gemm:ck")
        assert len(tasks) == 2

        # 2. Init GPU pool
        pool = GPUPool([0, 1])

        # 3. Init state
        store = StateStore(state_path)
        store.init_session("test-001", "mi355x", tasks, [0, 1])

        # 4. Allocate GPUs and add workers
        for t in tasks:
            task_id = f"{t['operator']}:{t['backend']}"
            gpu = pool.acquire(task_id)
            store.pop_task_queue()
            store.add_worker(task_id, gpu, f"opt/{task_id.replace(':', '-')}-001")

        state = store.load()
        assert len(state["workers"]) == 2
        assert state["task_queue"] == []

        # 5. Record rounds
        store.record_round("gemm:triton", 1, "persistent_kernel", 488.1, 614.2, "success")
        store.record_round("gemm:triton", 2, "block_m_tuning", 614.2, 710.5, "success")
        store.record_round("gemm:triton", 3, "xcd_swizzle", 710.5, 720.3, "success")

        # 6. Activity logging
        os.makedirs(os.path.dirname(activity_path), exist_ok=True)
        logger = ActivityLogger(activity_path, "gemm:triton")
        logger.log("VERIFY", 1, "benchmark complete: 614.2 TFLOPS (+25.8%)")

        entries = ActivityLogger.read_recent(activity_path)
        assert len(entries) == 1

        # 7. Bottleneck detection
        detector = BottleneckDetector(threshold=0.02, patience=3)
        state = store.load()
        rounds = state["workers"]["gemm:triton"]["rounds"]
        result = detector.check(rounds)
        assert result.is_bottleneck is False  # gains are still significant

        # 8. Simulate low-gain rounds triggering bottleneck
        store.record_round("gemm:triton", 4, "num_warps", 720.3, 725.1, "success")
        store.record_round("gemm:triton", 5, "async_copy", 725.1, 728.0, "success")
        store.record_round("gemm:triton", 6, "lds_padding", 728.0, 730.2, "success")

        state = store.load()
        rounds = state["workers"]["gemm:triton"]["rounds"]
        result = detector.check(rounds)
        assert result.is_bottleneck is True
        assert result.reason == "diminishing_returns"

        # 9. Update status
        store.update_worker_status("gemm:triton", WorkerStatus.BOTTLENECK, "L1")

        # 10. Complete and release GPU
        store.update_worker_status("gemm:triton", WorkerStatus.COMPLETED)
        pool.release("gemm:triton")
        assert pool.available_count() == 1

        # 11. Benchmark runner command generation
        runner = BenchmarkRunner(repo_root="/shared_nfs/yaoc/agent_work/Primus-Turbo", gpu_id=0)
        cmd = runner.build_benchmark_cmd("gemm", "fp8", "blockwise")
        assert "HIP_VISIBLE_DEVICES=0" in cmd
        assert "bench_gemm_turbo.py" in cmd
