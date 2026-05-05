from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
import random
import time

try:
    import psutil
except Exception:
    psutil = None


@dataclass
class DecisionPlan:
    split_id: str
    local_k: int
    use_cloud: bool
    score_s: float
    metrics: dict


class SplitDecision:
    """
    专利化动态切分决策引擎。

    对应专利步骤：
    - 步骤2：采集多维实时性能数据
    - 步骤3.1：加权移动平均 + 动态权重 -> 综合负载分数 S
    - 步骤3.2：D_type + S 查询二维映射表（SPLIT_00 ~ SPLIT_04）
    - 步骤5：反馈闭环记录与阈值微调
    """

    # SPLIT 标识到执行计划的映射（按你的要求硬编码）
    # 极高负载场景回退到纯云执行（local_k=0）
    SPLIT_EXECUTION_MAP = {
        "SPLIT_04": {"local_k": 15, "use_cloud": True},  # 低负载
        "SPLIT_03": {"local_k": 12, "use_cloud": True},  # 中低负载
        "SPLIT_02": {"local_k": 10, "use_cloud": True},  # 中负载
        "SPLIT_01": {"local_k": 8, "use_cloud": True},   # 高负载
        "SPLIT_00": {"local_k": 0, "use_cloud": True},   # 极高负载
    }

    # 专利表2（二维映射）: 每个 D_type 对应 4 个分段阈值
    # 分段规则：
    # S < b0 -> SPLIT_04
    # b0 <= S < b1 -> SPLIT_03
    # b1 <= S < b2 -> SPLIT_02
    # b2 <= S < b3 -> SPLIT_01
    # S >= b3 -> SPLIT_00
    DEFAULT_SPLIT_BOUNDARIES = {
        # MOD: 提高 b2（70 -> 80），让更多查询落入中低负载区间
        1: [30.0, 50.0, 80.0, 90.0],  # 高性能终端
        2: [30.0, 50.0, 80.0, 88.0],  # 通用移动终端
        3: [30.0, 50.0, 80.0, 85.0],  # 资源受限终端
    }

    def __init__(
        self,
        device_type: int = 2,
        history_window: int = 5,
        mem_threshold: float = 75.0,
        weight_step: float = 0.05,
    ):
        # 优先使用传入设备类型，非法值回退到通用移动终端 D_type=2
        self.device_type = int(device_type) if int(device_type) in {1, 2, 3} else 2
        self.history_window = max(3, int(history_window))
        self.mem_threshold = float(mem_threshold)
        self.weight_step = float(weight_step)

        # 步骤3.1：初始权重
        self.weight_cpu = 0.5
        self.weight_mem = 0.5

        # 时间序列（用于加权移动平均）
        self.cpu_history = deque(maxlen=self.history_window)
        self.mem_history = deque(maxlen=self.history_window)

        # 反馈闭环（步骤5）
        self.feedback_records: list[dict] = []
        self.feedback_update_interval = 10
        self.target_total_latency_ms = 2000.0

        # 可在线微调的阈值表
        self.split_boundaries = {
            d_type: list(bounds) for d_type, bounds in self.DEFAULT_SPLIT_BOUNDARIES.items()
        }

        # 兼容旧逻辑：把 split_id 映射回 0/1/2
        self.legacy_split_map = {
            "SPLIT_00": 0,
            "SPLIT_01": 1,
            "SPLIT_02": 2,
            "SPLIT_03": 2,
            "SPLIT_04": 2,
        }

    def get_system_load(self):
        """旧接口兼容：返回 (cpu_percent, mem_percent)。"""
        metrics = self.collect_runtime_metrics()
        return metrics["cpu_percent"], metrics["memory_percent"]

    def collect_runtime_metrics(self) -> dict:
        """
        专利步骤2：多维实时性能采集。
        - CPU 各核心利用率
        - 内存占用率
        - 可用带宽（可模拟）
        - RTT（可模拟）
        - 电量（可模拟）
        """
        if psutil is None:
            core_utils = [50.0, 50.0, 50.0, 50.0]
            cpu_percent = 50.0
            mem_percent = 50.0
            battery_percent = float(os.getenv("NEXUS_SIM_BATTERY_PERCENT", "60"))
        else:
            core_utils = psutil.cpu_percent(interval=0.1, percpu=True)
            cpu_percent = sum(core_utils) / len(core_utils) if core_utils else psutil.cpu_percent(interval=0.1)
            mem_percent = psutil.virtual_memory().percent

            battery_info = None
            try:
                battery_info = psutil.sensors_battery()
            except Exception:
                battery_info = None

            if battery_info is not None and battery_info.percent is not None:
                battery_percent = float(battery_info.percent)
            else:
                battery_percent = float(os.getenv("NEXUS_SIM_BATTERY_PERCENT", str(round(40 + random.random() * 50, 2))))

        bandwidth_mbps = float(os.getenv("NEXUS_SIM_BANDWIDTH_MBPS", str(round(20 + random.random() * 80, 2))))
        rtt_ms = float(os.getenv("NEXUS_SIM_RTT_MS", str(round(10 + random.random() * 90, 2))))

        metrics = {
            "timestamp": time.time(),
            "cpu_core_utils": [float(x) for x in core_utils],
            "cpu_percent": float(cpu_percent),
            "memory_percent": float(mem_percent),
            "bandwidth_mbps": float(bandwidth_mbps),
            "rtt_ms": float(rtt_ms),
            "battery_percent": float(battery_percent),
        }
        return metrics

    def _weighted_moving_average(self, values: list[float]) -> float:
        if not values:
            return 50.0

        # 近期权重大（线性递增权重）
        weights = list(range(1, len(values) + 1))
        weighted_sum = sum(v * w for v, w in zip(values, weights))
        return weighted_sum / sum(weights)

    def _dynamic_weight_adjustment(self, memory_avg: float) -> None:
        """
        专利3.1：动态权重调整。
        - 若内存均值超过阈值，增加 W_m，减少 W_c（步长 delta=0.05）
        """
        if memory_avg > self.mem_threshold:
            self.weight_mem = min(0.8, self.weight_mem + self.weight_step)
            self.weight_cpu = max(0.2, self.weight_cpu - self.weight_step)
        else:
            # 当内存压力下降时，缓慢回归均衡权重
            self.weight_mem = max(0.35, self.weight_mem - self.weight_step / 2)
            self.weight_cpu = min(0.65, self.weight_cpu + self.weight_step / 2)

    def update_histories(self, cpu_percent: float, memory_percent: float) -> None:
        self.cpu_history.append(float(cpu_percent))
        self.mem_history.append(float(memory_percent))

    def compute_load_score(self, metrics: dict) -> tuple[float, float, float]:
        """
        专利3.1公式：
        S = W_c * C_bar + W_m * M_bar
        返回 (S, C_bar, M_bar)
        """
        self.update_histories(metrics["cpu_percent"], metrics["memory_percent"])

        c_bar = self._weighted_moving_average(list(self.cpu_history))
        m_bar = self._weighted_moving_average(list(self.mem_history))

        self._dynamic_weight_adjustment(m_bar)
        score_s = (self.weight_cpu * c_bar) + (self.weight_mem * m_bar)
        score_s = max(0.0, min(100.0, float(score_s)))
        return score_s, c_bar, m_bar

    def lookup_split_id(self, score_s: float, d_type: int | None = None) -> str:
        """专利表2：二维映射查表。"""
        device_type = d_type if d_type in {1, 2, 3} else self.device_type
        b0, b1, b2, b3 = self.split_boundaries[device_type]

        if score_s < b0:
            return "SPLIT_04"
        if score_s < b1:
            return "SPLIT_03"
        if score_s < b2:
            return "SPLIT_02"
        if score_s < b3:
            return "SPLIT_01"
        return "SPLIT_00"

    def split_to_plan(self, split_id: str, score_s: float, metrics: dict) -> DecisionPlan:
        cfg = self.SPLIT_EXECUTION_MAP.get(split_id, self.SPLIT_EXECUTION_MAP["SPLIT_00"])
        return DecisionPlan(
            split_id=split_id,
            local_k=int(cfg["local_k"]),
            use_cloud=bool(cfg["use_cloud"]),
            score_s=float(score_s),
            metrics=metrics,
        )

    def decide_plan(self, forced_split_id: str | None = None) -> DecisionPlan:
        metrics = self.collect_runtime_metrics()
        score_s, c_bar, m_bar = self.compute_load_score(metrics)
        metrics["cpu_wma"] = round(c_bar, 4)
        metrics["mem_wma"] = round(m_bar, 4)
        metrics["weight_cpu"] = round(self.weight_cpu, 4)
        metrics["weight_mem"] = round(self.weight_mem, 4)

        split_id = forced_split_id if forced_split_id else self.lookup_split_id(score_s, self.device_type)
        return self.split_to_plan(split_id=split_id, score_s=score_s, metrics=metrics)

    def decide(self, force_level=None):
        """
        旧接口兼容：返回 0/1/2。
        - force_level 允许强制指定 legacy split point（0/1/2）
        """
        if force_level is not None:
            try:
                force_level = int(force_level)
            except Exception:
                force_level = None
            if force_level in {0, 1, 2}:
                return force_level

        plan = self.decide_plan()
        return self.legacy_split_map.get(plan.split_id, 1)

    def record_feedback(
        self,
        edge_compute_ms: float,
        network_ms: float,
        cloud_compute_ms: float,
        total_ms: float,
        energy_delta: float,
    ) -> None:
        """专利步骤5：闭环反馈记录。"""
        self.feedback_records.append(
            {
                "timestamp": time.time(),
                "edge_compute_ms": float(edge_compute_ms),
                "network_ms": float(network_ms),
                "cloud_compute_ms": float(cloud_compute_ms),
                "total_ms": float(total_ms),
                "energy_delta": float(energy_delta),
            }
        )

        if len(self.feedback_records) % self.feedback_update_interval == 0:
            self._update_boundaries_with_feedback()

    def _update_boundaries_with_feedback(self) -> None:
        """
        每10次查询做一次阈值微调：
        - 若最近平均总时延偏高：降低阈值，促使更多请求走低 local_k（更偏云端）
        - 若最近平均总时延偏低：提高阈值，鼓励端侧多承担
        """
        recent = self.feedback_records[-self.feedback_update_interval :]
        if not recent:
            return

        avg_total = sum(item["total_ms"] for item in recent) / len(recent)
        bounds = self.split_boundaries[self.device_type]

        if avg_total > self.target_total_latency_ms * 1.1:
            delta = -2.0
        elif avg_total < self.target_total_latency_ms * 0.8:
            delta = 2.0
        else:
            delta = 0.0

        if delta == 0.0:
            return

        new_bounds = [max(5.0, min(95.0, b + delta)) for b in bounds]
        new_bounds.sort()

        # 保证边界严格递增，避免重叠
        for i in range(1, len(new_bounds)):
            if new_bounds[i] <= new_bounds[i - 1]:
                new_bounds[i] = min(95.0, new_bounds[i - 1] + 1.0)

        self.split_boundaries[self.device_type] = new_bounds

    # Backward-compatible alias for previous call sites.
    def decide_split_point(self, force_level=None):
        return self.decide(force_level=force_level)


# MOD: 默认实例显式使用高性能设备类型
split_decision = SplitDecision(device_type=1)