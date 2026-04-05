import hashlib
import json
import time
import uuid

from flask import Blueprint, jsonify, render_template, request

from app.models import ABResult, PolicyDecision, PredictionOutput, RuntimeFeature, UserHistory, db
from app.services.llm_service import query_llm_api
from app.services.policy_service import choose_policy, get_latest_policy, reload_policy
from app.services.predictor_service import predict_state_and_label
from app.services.privacy_service import encrypt_text, mask_sensitive_text, privacy_meta
from app.services.retrieval_service import retrieve_content
from app.services.risk_service import evaluate_risk
from app.services.weighting_service import update_weights

bp = Blueprint('routes', __name__)


def _query_hash(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _avg_doc_score(docs) -> float:
    if not docs:
        return 0.0
    total = sum(float(d.get("score", 0.0)) for d in docs)
    return total / len(docs)


def _build_feature_window(user_id: int, window_size: int = 5):
    rows = (
        RuntimeFeature.query.filter_by(user_id=user_id)
        .order_by(RuntimeFeature.id.desc())
        .limit(window_size)
        .all()
    )
    rows.reverse()
    return [{"risk_score": float(r.risk_score)} for r in rows]


def _safe_json_loads(payload: str):
    try:
        return json.loads(payload or "{}")
    except Exception:
        return {}

@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json(silent=True) or {}
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        question = (data.get('question') or '').strip()
        user_id = int(data.get('user_id') or 1)
        debug = bool(data.get('debug') or False)
        request_id = uuid.uuid4().hex
        started_at = time.perf_counter()

        if not question:
            return jsonify({'error': 'Question is required'}), 400

        # First-pass retrieval for risk features.
        seed_docs = retrieve_content(question)
        seed_avg_score = _avg_doc_score(seed_docs)
        elapsed_seed = (time.perf_counter() - started_at) * 1000.0

        risk = evaluate_risk(
            {
                "query_length": len(question),
                "retrieved_docs": len(seed_docs),
                "avg_doc_score": seed_avg_score,
                "latency_ms": elapsed_seed,
            }
        )

        last_policy = (
            PolicyDecision.query.filter_by(user_id=user_id)
            .order_by(PolicyDecision.id.desc())
            .first()
        )
        prev_weights = None
        if last_policy:
            prev_weights = {"Wc": float(last_policy.weight_c), "Wm": float(last_policy.weight_m)}

        weights = update_weights(
            risk_level=risk["level"],
            context_signal=1.0 - float(risk.get("sparsity", 0.0)),
            memory_signal=float(risk.get("score", 0.0)),
            prev_weights=prev_weights,
        )

        feature_window = _build_feature_window(user_id=user_id)
        prediction = predict_state_and_label(feature_window, current_score=float(risk["score"]))
        policy = choose_policy(
            predicted_label=str(prediction["predicted_label"]),
            predicted_score=float(prediction["predicted_score"]),
        )

        # Second-pass retrieval with strategy policy.
        retrieved_docs = retrieve_content(question, policy["retrieval_config"])

        # Privacy masking for prompt safety.
        masked_question = mask_sensitive_text(question)
        meta = privacy_meta(question, masked_question)

        context = ""
        if retrieved_docs:
            context = "\n".join([f"标题: {d['title']}\n内容: {d['content']}" for d in retrieved_docs])

        prompt = (
            "请基于以下内容库回答问题，并避免输出任何可能的个人敏感信息。\n\n"
            f"内容库：\n{context}\n\n"
            f"问题：{masked_question}\n"
            "回答："
        )

        answer = query_llm_api(prompt)
        safe_answer = mask_sensitive_text(answer)

        latency_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
        query_hash = _query_hash(question)
        cohort = "A" if int(query_hash[:2], 16) % 2 == 0 else "B"
        quality_proxy = min(1.0, len(safe_answer or "") / max(80.0, float(len(question) * 6)))

        encrypted_q = encrypt_text(question)
        encrypted_a = encrypt_text(safe_answer)

        history = UserHistory(user_id=user_id, question=encrypted_q, answer=encrypted_a)
        runtime = RuntimeFeature(
            request_id=request_id,
            user_id=user_id,
            query_hash=query_hash,
            query_length=len(question),
            retrieved_docs=len(retrieved_docs),
            avg_doc_score=round(_avg_doc_score(retrieved_docs), 4),
            risk_level=str(risk["level"]),
            risk_score=float(risk["score"]),
            latency_ms=latency_ms,
        )
        decision = PolicyDecision(
            request_id=request_id,
            user_id=user_id,
            policy_version=int(policy["policy_version"]),
            strategy_label=str(policy["strategy_label"]),
            weight_c=float(weights["Wc"]),
            weight_m=float(weights["Wm"]),
            retrieval_config=json.dumps(policy["retrieval_config"], ensure_ascii=False),
        )
        pred = PredictionOutput(
            request_id=request_id,
            user_id=user_id,
            predicted_score=float(prediction["predicted_score"]),
            predicted_label=str(prediction["predicted_label"]),
            confidence=float(prediction["confidence"]),
            window_size=int(prediction["window_size"]),
        )
        ab = ABResult(
            request_id=request_id,
            cohort=cohort,
            metric_name="quality_proxy",
            metric_value=round(quality_proxy, 4),
        )

        db.session.add_all([history, runtime, decision, pred, ab])
        db.session.commit()

        payload = {
            'request_id': request_id,
            'response': safe_answer,
        }
        if debug:
            payload['debug'] = {
                'risk': risk,
                'weights': weights,
                'prediction': prediction,
                'policy': policy,
                'privacy_meta': meta,
                'latency_ms': latency_ms,
                'retrieved_docs': len(retrieved_docs),
            }

        return jsonify(payload), 200
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        db.session.rollback()
        return jsonify({'error': str(e), 'trace': error_msg}), 500


@bp.route('/policy/latest', methods=['GET'])
def policy_latest():
    latest = get_latest_policy()
    decision = PolicyDecision.query.order_by(PolicyDecision.id.desc()).first()

    latest_decision = None
    if decision:
        latest_decision = {
            'request_id': decision.request_id,
            'user_id': decision.user_id,
            'policy_version': decision.policy_version,
            'strategy_label': decision.strategy_label,
            'weights': {'Wc': decision.weight_c, 'Wm': decision.weight_m},
            'retrieval_config': _safe_json_loads(decision.retrieval_config),
            'created_at': str(decision.created_at),
        }

    return jsonify({'policy': latest, 'latest_decision': latest_decision}), 200


@bp.route('/policy/reload', methods=['POST'])
def policy_reload():
    data = request.get_json(silent=True) or {}
    mapping = data.get('map') if isinstance(data.get('map'), dict) else None
    updated = reload_policy(mapping)
    return jsonify({'message': 'policy reloaded', 'policy': updated}), 200


@bp.route('/metrics/runtime', methods=['GET'])
def runtime_metrics():
    n = int(request.args.get('n', 50) or 50)
    n = max(1, min(500, n))

    rows = RuntimeFeature.query.order_by(RuntimeFeature.id.desc()).limit(n).all()
    if not rows:
        return jsonify(
            {
                'sample_size': 0,
                'avg_latency_ms': 0.0,
                'avg_risk_score': 0.0,
                'risk_distribution': {'P_high': 0, 'P_medium': 0, 'P_low': 0},
                'avg_quality_proxy': 0.0,
            }
        ), 200

    sample_size = len(rows)
    avg_latency = round(sum(float(r.latency_ms) for r in rows) / sample_size, 2)
    avg_risk = round(sum(float(r.risk_score) for r in rows) / sample_size, 4)

    dist = {'P_high': 0, 'P_medium': 0, 'P_low': 0}
    for r in rows:
        dist[r.risk_level] = dist.get(r.risk_level, 0) + 1

    ab_rows = ABResult.query.order_by(ABResult.id.desc()).limit(n).all()
    quality_values = [float(a.metric_value) for a in ab_rows if a.metric_name == 'quality_proxy']
    avg_quality = round(sum(quality_values) / len(quality_values), 4) if quality_values else 0.0

    return jsonify(
        {
            'sample_size': sample_size,
            'avg_latency_ms': avg_latency,
            'avg_risk_score': avg_risk,
            'risk_distribution': dist,
            'avg_quality_proxy': avg_quality,
        }
    ), 200