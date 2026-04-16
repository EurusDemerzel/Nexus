from app.services.split_decision import SplitDecision


def test_compute_load_score_and_dynamic_weights():
    engine = SplitDecision(device_type=2, history_window=4, mem_threshold=60.0, weight_step=0.05)

    # 先喂入中等负载
    s1, c1, m1 = engine.compute_load_score({"cpu_percent": 40.0, "memory_percent": 50.0})
    assert 0.0 <= s1 <= 100.0
    assert c1 > 0.0
    assert m1 > 0.0

    # 连续高内存，触发动态权重调整（W_m 应上升）
    engine.compute_load_score({"cpu_percent": 42.0, "memory_percent": 88.0})
    engine.compute_load_score({"cpu_percent": 45.0, "memory_percent": 92.0})

    assert engine.weight_mem >= 0.5
    assert engine.weight_mem >= engine.weight_cpu


def test_split_lookup_mapping_for_device_type_2():
    engine = SplitDecision(device_type=2)
    engine.split_boundaries[2] = [20.0, 40.0, 60.0, 80.0]

    assert engine.lookup_split_id(10.0, d_type=2) == "SPLIT_04"
    assert engine.lookup_split_id(30.0, d_type=2) == "SPLIT_03"
    assert engine.lookup_split_id(50.0, d_type=2) == "SPLIT_02"
    assert engine.lookup_split_id(70.0, d_type=2) == "SPLIT_01"
    assert engine.lookup_split_id(90.0, d_type=2) == "SPLIT_00"


def test_split_execution_plan_constants():
    engine = SplitDecision(device_type=2)
    plan = engine.split_to_plan(
        split_id="SPLIT_00",
        score_s=95.0,
        metrics={"cpu_percent": 90.0, "memory_percent": 90.0},
    )

    assert plan.local_k == 0
    assert plan.use_cloud is True
    assert plan.split_id == "SPLIT_00"
