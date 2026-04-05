from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Patent(db.Model):
    __tablename__ = 'patents'
    
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), nullable=False)
    content = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    
    def __repr__(self):
        return f'<Patent {self.title}>'

class UserHistory(db.Model):
    __tablename__ = 'user_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
    
    def __repr__(self):
        return f'<UserHistory {self.question}>'


class RuntimeFeature(db.Model):
    __tablename__ = 'runtime_features'

    id = db.Column(db.Integer, primary_key=True)
    request_id = db.Column(db.String(64), index=True, nullable=False)
    user_id = db.Column(db.Integer, nullable=False, default=1)
    query_hash = db.Column(db.String(64), nullable=False)
    query_length = db.Column(db.Integer, nullable=False, default=0)
    retrieved_docs = db.Column(db.Integer, nullable=False, default=0)
    avg_doc_score = db.Column(db.Float, nullable=False, default=0.0)
    risk_level = db.Column(db.String(20), nullable=False, default='P_medium')
    risk_score = db.Column(db.Float, nullable=False, default=0.0)
    latency_ms = db.Column(db.Float, nullable=False, default=0.0)
    created_at = db.Column(db.DateTime, server_default=db.func.now())


class PolicyDecision(db.Model):
    __tablename__ = 'policy_decisions'

    id = db.Column(db.Integer, primary_key=True)
    request_id = db.Column(db.String(64), index=True, nullable=False)
    user_id = db.Column(db.Integer, nullable=False, default=1)
    policy_version = db.Column(db.Integer, nullable=False, default=1)
    strategy_label = db.Column(db.String(40), nullable=False)
    weight_c = db.Column(db.Float, nullable=False, default=0.5)
    weight_m = db.Column(db.Float, nullable=False, default=0.5)
    retrieval_config = db.Column(db.Text, nullable=False, default='{}')
    created_at = db.Column(db.DateTime, server_default=db.func.now())


class PredictionOutput(db.Model):
    __tablename__ = 'prediction_outputs'

    id = db.Column(db.Integer, primary_key=True)
    request_id = db.Column(db.String(64), index=True, nullable=False)
    user_id = db.Column(db.Integer, nullable=False, default=1)
    predicted_score = db.Column(db.Float, nullable=False, default=0.0)
    predicted_label = db.Column(db.String(40), nullable=False, default='balanced')
    confidence = db.Column(db.Float, nullable=False, default=0.0)
    window_size = db.Column(db.Integer, nullable=False, default=0)
    created_at = db.Column(db.DateTime, server_default=db.func.now())


class ABResult(db.Model):
    __tablename__ = 'ab_results'

    id = db.Column(db.Integer, primary_key=True)
    request_id = db.Column(db.String(64), index=True, nullable=False)
    cohort = db.Column(db.String(20), nullable=False, default='A')
    metric_name = db.Column(db.String(64), nullable=False)
    metric_value = db.Column(db.Float, nullable=False, default=0.0)
    created_at = db.Column(db.DateTime, server_default=db.func.now())