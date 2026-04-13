import argparse
import json
import sys
from pathlib import Path

from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.app import create_app
from app.models import Material, MaterialEmbedding, db
from app.services.embedding_service import embed_text


def is_valid_content(content: str) -> bool:
    text = (content or "").strip()
    if not text:
        return False
    if text.startswith("[文件:"):
        return False
    return len(text) >= 50


def upsert_embedding(material: Material) -> bool:
    vec = embed_text(material.content or "")
    vector_list = [float(x) for x in vec]

    if not vector_list:
        return False

    row = MaterialEmbedding.query.filter_by(material_id=material.id).first()
    if row is None:
        row = MaterialEmbedding(
            material_id=material.id,
            embedding_json=json.dumps(vector_list, ensure_ascii=False),
            dimension=len(vector_list),
        )
        db.session.add(row)
    else:
        row.embedding_json = json.dumps(vector_list, ensure_ascii=False)
        row.dimension = len(vector_list)

    return True


def ingest(limit: int | None = None, rebuild: bool = False) -> None:
    app = create_app()
    with app.app_context():
        if rebuild:
            db.session.execute(text("DROP TABLE IF EXISTS material_embeddings"))
            db.session.commit()
            MaterialEmbedding.__table__.create(bind=db.engine, checkfirst=True)

        query = Material.query
        if limit is not None and limit > 0:
            query = query.limit(limit)

        materials = query.all()
        total = len(materials)
        valid = 0
        written = 0

        for item in materials:
            if not is_valid_content(item.content or ""):
                continue
            valid += 1
            if upsert_embedding(item):
                written += 1

        db.session.commit()
        print(f"Ingest done. total={total}, valid={valid}, embedded={written}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vector rows for materials table")
    parser.add_argument("--limit", type=int, default=None, help="Only ingest first N materials")
    parser.add_argument("--rebuild", action="store_true", help="Delete and rebuild all material_embeddings")
    args = parser.parse_args()
    ingest(limit=args.limit, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
