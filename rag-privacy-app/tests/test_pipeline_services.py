from app.services.policy_service import choose_policy
from app.services.predictor_service import predict_state_and_label
from app.services.risk_service import evaluate_risk
from app.services.weighting_service import update_weights


def test_risk_level_outputs_are_valid():
    result = evaluate_risk(
        {
            "query_length": 64,
            "retrieved_docs": 1,
            "avg_doc_score": 1,
            "latency_ms": 2000,
        }
    )
    assert result["level"] in {"P_high", "P_medium", "P_low"}
    assert 0.0 <= float(result["score"]) <= 1.0


def test_weight_update_is_normalized_and_bounded():
    weights = update_weights(
        risk_level="P_high",
        context_signal=0.8,
        memory_signal=0.9,
        prev_weights={"Wc": 0.55, "Wm": 0.45},
    )
    s = round(weights["Wc"] + weights["Wm"], 4)
    assert s == 1.0
    assert 0.1 <= weights["Wc"] <= 0.9
    assert 0.1 <= weights["Wm"] <= 0.9


def test_predictor_returns_label_and_score():
    pred = predict_state_and_label(
        [{"risk_score": 0.2}, {"risk_score": 0.4}, {"risk_score": 0.7}],
        current_score=0.6,
    )
    assert pred["predicted_label"] in {"precision_first", "balanced", "recall_first"}
    assert 0.0 <= float(pred["predicted_score"]) <= 1.0


def test_policy_mapping_contains_retrieval_config():
    policy = choose_policy("balanced", 0.5)
    cfg = policy["retrieval_config"]
    assert "candidate_limit" in cfg
    assert "chunk_limit" in cfg
    assert cfg["chunk_limit"] > 0
