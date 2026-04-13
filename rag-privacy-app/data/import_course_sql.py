import argparse
import datetime as dt
import os
from pathlib import Path

import pymysql
from dotenv import load_dotenv
from sqlalchemy.engine.url import make_url


def _split_sql_statements(sql_text: str) -> list[str]:
    statements: list[str] = []
    buf: list[str] = []
    in_single = False
    in_double = False
    escape = False

    for ch in sql_text:
        if escape:
            buf.append(ch)
            escape = False
            continue

        if ch == "\\":
            buf.append(ch)
            escape = True
            continue

        if ch == "'" and not in_double:
            in_single = not in_single
            buf.append(ch)
            continue

        if ch == '"' and not in_single:
            in_double = not in_double
            buf.append(ch)
            continue

        if ch == ";" and not in_single and not in_double:
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
            continue

        buf.append(ch)

    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)

    return statements


def _normalize_dump(sql_text: str) -> str:
    cleaned_lines = []
    for line in sql_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("--"):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def _should_skip(statement_upper: str) -> bool:
    if statement_upper.startswith("SET @@GLOBAL.GTID_PURGED"):
        return True
    if statement_upper.startswith("CREATE DATABASE"):
        return True
    if statement_upper.startswith("USE "):
        return True
    if statement_upper.startswith("DROP TABLE"):
        return True
    return False


def _backup_table(cursor, table_name: str, suffix: str) -> None:
    backup_table = f"{table_name}_bak_{suffix}"
    cursor.execute(f"DROP TABLE IF EXISTS {backup_table}")
    cursor.execute(f"CREATE TABLE {backup_table} AS SELECT * FROM {table_name}")


def _table_exists(cursor, table_name: str) -> bool:
    cursor.execute(
        """
        SELECT COUNT(*)
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s
        """,
        (table_name,),
    )
    return int(cursor.fetchone()[0] or 0) > 0


def _drop_target_tables(cursor) -> None:
    cursor.execute("SET FOREIGN_KEY_CHECKS=0")
    cursor.execute("DROP TABLE IF EXISTS material_embeddings")
    cursor.execute("DROP TABLE IF EXISTS materials")
    cursor.execute("DROP TABLE IF EXISTS courses")
    cursor.execute("SET FOREIGN_KEY_CHECKS=1")


def import_dump(sql_file: Path, rebuild: bool, backup: bool, insert_ignore: bool) -> None:
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set in environment or .env")

    parsed = make_url(db_url)
    connection = pymysql.connect(
        host=parsed.host or "localhost",
        port=parsed.port or 3306,
        user=parsed.username,
        password=parsed.password,
        database=parsed.database,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.Cursor,
        autocommit=False,
    )

    raw_sql = sql_file.read_text(encoding="utf-8", errors="ignore")
    normalized_sql = _normalize_dump(raw_sql)
    statements = _split_sql_statements(normalized_sql)

    executed = 0
    skipped = 0
    suffix = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        with connection.cursor() as cursor:
            if rebuild:
                if backup:
                    if _table_exists(cursor, "courses"):
                        _backup_table(cursor, "courses", suffix)
                    if _table_exists(cursor, "materials"):
                        _backup_table(cursor, "materials", suffix)
                _drop_target_tables(cursor)

            for stmt in statements:
                stmt_upper = stmt.strip().upper()
                if _should_skip(stmt_upper):
                    skipped += 1
                    continue

                if insert_ignore and stmt_upper.startswith("INSERT INTO"):
                    stmt = "INSERT IGNORE INTO" + stmt[len("INSERT INTO") :]

                cursor.execute(stmt)
                executed += 1

            cursor.execute("SELECT COUNT(*) FROM courses")
            courses_count = int(cursor.fetchone()[0] or 0)
            cursor.execute("SELECT COUNT(*) FROM materials")
            materials_count = int(cursor.fetchone()[0] or 0)

        connection.commit()
    finally:
        connection.close()

    print(f"Import done. executed={executed}, skipped={skipped}")
    print(f"courses={courses_count}, materials={materials_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Import course_knowledge.sql into DATABASE_URL database")
    parser.add_argument("--sql-file", default="data/course_knowledge.sql", help="Path to SQL dump file")
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="Do not drop and recreate courses/materials before import",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Backup existing courses/materials tables before rebuild",
    )
    parser.add_argument(
        "--insert-ignore",
        action="store_true",
        help="Convert INSERT INTO to INSERT IGNORE INTO",
    )
    args = parser.parse_args()

    sql_path = Path(args.sql_file)
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    import_dump(
        sql_file=sql_path,
        rebuild=not args.no_rebuild,
        backup=args.backup,
        insert_ignore=args.insert_ignore,
    )


if __name__ == "__main__":
    main()
