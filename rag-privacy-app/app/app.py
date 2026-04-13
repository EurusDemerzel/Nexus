from flask import Flask
from app.routes import bp as main_routes
from app.models import db

import os

from sqlalchemy import text

def create_app():
    # Get the directory where app.py is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # The templates folder is one level up from app.py
    template_dir = os.path.join(os.path.dirname(current_dir), 'templates')
    
    app = Flask(__name__, template_folder=template_dir)
    
    # Force set config from env if config.py loading is tricky
    import dotenv
    dotenv.load_dotenv()
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key')

    db.init_app(app)

    # Register blueprints
    app.register_blueprint(main_routes)

    # Dev-friendly schema sync: create tables and add missing columns.
    # Note: db.create_all() will NOT add columns to an existing table.
    with app.app_context():
        db.create_all()

        def _ensure_column(table_name: str, column_name: str, alter_sql: str) -> None:
            try:
                query = text(
                    """
                    SELECT COUNT(*)
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_SCHEMA = DATABASE()
                      AND TABLE_NAME = :table
                      AND COLUMN_NAME = :col
                    """
                )
                count = db.session.execute(query, {"table": table_name, "col": column_name}).scalar()
                if int(count or 0) == 0:
                    db.session.execute(text(alter_sql))
                    db.session.commit()
            except Exception:
                db.session.rollback()

        _ensure_column(
            "patents",
            "created_at",
            "ALTER TABLE patents ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP",
        )
        _ensure_column(
            "user_history",
            "created_at",
            "ALTER TABLE user_history ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP",
        )
        _ensure_column(
            "runtime_features",
            "request_id",
            "ALTER TABLE runtime_features ADD COLUMN request_id VARCHAR(64) NOT NULL DEFAULT ''",
        )
        _ensure_column(
            "runtime_features",
            "created_at",
            "ALTER TABLE runtime_features ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP",
        )
        _ensure_column(
            "policy_decisions",
            "request_id",
            "ALTER TABLE policy_decisions ADD COLUMN request_id VARCHAR(64) NOT NULL DEFAULT ''",
        )
        _ensure_column(
            "policy_decisions",
            "created_at",
            "ALTER TABLE policy_decisions ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP",
        )
        _ensure_column(
            "prediction_outputs",
            "request_id",
            "ALTER TABLE prediction_outputs ADD COLUMN request_id VARCHAR(64) NOT NULL DEFAULT ''",
        )
        _ensure_column(
            "prediction_outputs",
            "created_at",
            "ALTER TABLE prediction_outputs ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP",
        )
        _ensure_column(
            "ab_results",
            "request_id",
            "ALTER TABLE ab_results ADD COLUMN request_id VARCHAR(64) NOT NULL DEFAULT ''",
        )
        _ensure_column(
            "ab_results",
            "created_at",
            "ALTER TABLE ab_results ADD COLUMN created_at DATETIME DEFAULT CURRENT_TIMESTAMP",
        )
        _ensure_column(
            "material_embeddings",
            "dimension",
            "ALTER TABLE material_embeddings ADD COLUMN dimension INT NOT NULL DEFAULT 0",
        )

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)