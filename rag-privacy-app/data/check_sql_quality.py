import argparse
import html
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

HTML_TAG_RE = re.compile(r"<[^>]+>")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
WHITESPACE_RE = re.compile(r"\s+")

TEXT_KEYWORDS = (
    "question",
    "description",
    "answer",
    "content",
    "title",
    "body",
    "summary",
    "remark",
)


def split_sql_statements(sql_text: str) -> List[str]:
    statements: List[str] = []
    buf: List[str] = []
    in_single = False
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

        if ch == "'":
            in_single = not in_single
            buf.append(ch)
            continue

        if ch == ";" and not in_single:
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


def parse_create_tables(statements: List[str]) -> Dict[str, List[str]]:
    schema: Dict[str, List[str]] = {}
    for stmt in statements:
        m = re.match(r"\s*CREATE\s+TABLE\s+`?(\w+)`?\s*\((.*)\)\s*$", stmt, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            continue

        table = m.group(1)
        body = m.group(2)
        cols: List[str] = []
        for line in body.splitlines():
            raw = line.strip().rstrip(",")
            if not raw:
                continue
            upper = raw.upper()
            if upper.startswith(("PRIMARY KEY", "KEY ", "UNIQUE", "CONSTRAINT", "FULLTEXT", "INDEX", "FOREIGN KEY")):
                continue
            col_match = re.match(r"`?(\w+)`?\s+", raw)
            if col_match:
                cols.append(col_match.group(1))
        if cols:
            schema[table] = cols
    return schema


def split_insert_rows(values_sql: str) -> List[str]:
    rows: List[str] = []
    buf: List[str] = []
    in_single = False
    escape = False
    depth = 0

    for ch in values_sql:
        if escape:
            buf.append(ch)
            escape = False
            continue

        if ch == "\\":
            buf.append(ch)
            escape = True
            continue

        if ch == "'":
            in_single = not in_single
            buf.append(ch)
            continue

        if not in_single and ch == "(":
            depth += 1
            if depth == 1:
                buf = []
                continue

        if not in_single and ch == ")":
            depth -= 1
            if depth == 0:
                rows.append("".join(buf))
                buf = []
                continue

        if depth >= 1:
            buf.append(ch)

    return rows


def split_row_fields(row_sql: str) -> List[str]:
    parts: List[str] = []
    buf: List[str] = []
    in_single = False
    escape = False

    for ch in row_sql:
        if escape:
            buf.append(ch)
            escape = False
            continue

        if ch == "\\":
            buf.append(ch)
            escape = True
            continue

        if ch == "'":
            in_single = not in_single
            buf.append(ch)
            continue

        if ch == "," and not in_single:
            parts.append("".join(buf).strip())
            buf = []
            continue

        buf.append(ch)

    parts.append("".join(buf).strip())
    return parts


def decode_sql_string(token: str) -> str:
    inner = token[1:-1]
    inner = inner.replace("\\'", "'")
    inner = inner.replace('\\"', '"')
    inner = inner.replace("\\n", "\n")
    inner = inner.replace("\\r", "\r")
    inner = inner.replace("\\t", "\t")
    inner = inner.replace("\\\\", "\\")
    return inner


def parse_literal(token: str) -> Tuple[object, bool]:
    t = token.strip()
    if not t:
        return "", False

    if t.upper() == "NULL":
        return None, False

    if len(t) >= 2 and t[0] == "'" and t[-1] == "'":
        return decode_sql_string(t), False

    if re.fullmatch(r"-?\d+(\.\d+)?", t):
        return t, False

    # Suspicious unquoted literal, often SQL issue for varchar values.
    if re.fullmatch(r"[A-Za-z0-9_\-]+", t):
        return t, True

    return t, False


def detect_noise(text: str) -> List[str]:
    issues: List[str] = []
    if not text:
        issues.append("empty")
        return issues

    if HTML_TAG_RE.search(text):
        issues.append("html_tag")

    if "\ufffd" in text:
        issues.append("garbled_char")

    if "sql syntax" in text.lower() or "you have an error in your sql" in text.lower():
        issues.append("sql_error_text")

    if len(text.strip()) < 10:
        issues.append("too_short")

    return issues


def clean_text(text: str) -> str:
    out = html.unescape(text or "")
    out = HTML_TAG_RE.sub(" ", out)
    out = CONTROL_RE.sub(" ", out)
    out = WHITESPACE_RE.sub(" ", out)
    return out.strip()


def choose_text(record: Dict[str, object]) -> str:
    text_candidates: List[str] = []
    for key, value in record.items():
        if value is None:
            continue
        if not isinstance(value, str):
            continue
        key_lower = key.lower()
        if any(k in key_lower for k in TEXT_KEYWORDS):
            text_candidates.append(value)

    if text_candidates:
        return "\n".join(text_candidates)

    fallback = [str(v) for v in record.values() if isinstance(v, str) and len(v.strip()) >= 10]
    return "\n".join(fallback)


def analyze_sql(sql_path: Path) -> Dict[str, object]:
    sql_text = sql_path.read_text(encoding="utf-8", errors="ignore")
    statements = split_sql_statements(sql_text)

    create_tables = parse_create_tables(statements)
    table_names = sorted(create_tables.keys())

    insert_statements = []
    parsed_rows = []
    suspicious_literal_count = 0

    for stmt in statements:
        m = re.match(
            r"\s*INSERT\s+INTO\s+`?(\w+)`?\s*(?:\((.*?)\))?\s*VALUES\s*(.*)\s*$",
            stmt,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not m:
            continue

        insert_statements.append(stmt)
        table = m.group(1)
        col_sql = m.group(2)
        values_sql = m.group(3)

        if col_sql:
            columns = [c.strip().strip("`") for c in col_sql.split(",")]
        else:
            columns = create_tables.get(table, [])

        rows = split_insert_rows(values_sql)
        for row in rows:
            tokens = split_row_fields(row)
            literals = []
            row_suspicious = False
            for token in tokens:
                value, suspicious = parse_literal(token)
                if suspicious:
                    row_suspicious = True
                    suspicious_literal_count += 1
                literals.append(value)

            if columns and len(columns) == len(literals):
                record = dict(zip(columns, literals))
            else:
                record = {f"col_{i}": v for i, v in enumerate(literals)}

            parsed_rows.append({
                "table": table,
                "raw": row,
                "record": record,
                "suspicious_literal": row_suspicious,
            })

    total_records = len(parsed_rows)

    sample_size = min(3, total_records)
    sample_rows = random.sample(parsed_rows, sample_size) if sample_size else []

    sample_findings = []
    for item in sample_rows:
        raw_preview = item["raw"][:300]
        issues = detect_noise(raw_preview)
        if item["suspicious_literal"]:
            issues.append("unquoted_literal")
        sample_findings.append(
            {
                "table": item["table"],
                "preview": raw_preview,
                "issues": sorted(set(issues)) if issues else ["none"],
            }
        )

    # Build cleaned records for RAG usage.
    cleaned_records = []
    for item in parsed_rows:
        rec = item["record"]
        text = choose_text(rec)
        text = clean_text(text)
        if len(text) < 10:
            continue
        cleaned_records.append(
            {
                "table": item["table"],
                "id": rec.get("id"),
                "text": text,
                "metadata": {k: rec.get(k) for k in rec.keys() if k != "content"},
            }
        )

    has_noise = False
    reasons = []
    if suspicious_literal_count > 0:
        has_noise = True
        reasons.append("发现未加引号的字面量，存在 SQL 语法/类型风险")

    sample_issue_count = sum(1 for x in sample_findings if x["issues"] != ["none"])
    if sample_issue_count >= 1:
        has_noise = True
        reasons.append("抽样数据含噪声特征（如乱码/过短/可疑字面量）")

    if len(cleaned_records) < max(1, int(total_records * 0.5)):
        has_noise = True
        reasons.append("可用文本占比较低，建议先清洗后再向量化")

    suitability = "适合直接向量化" if not has_noise else "不建议直接向量化，需先清洗"

    report = {
        "sql_file": str(sql_path),
        "table_count": len(table_names),
        "tables": table_names,
        "insert_statement_count": len(insert_statements),
        "total_records": total_records,
        "suspicious_literal_count": suspicious_literal_count,
        "sample_findings": sample_findings,
        "suitability": suitability,
        "reasons": reasons,
        "cleaned_record_count": len(cleaned_records),
    }

    return {
        "report": report,
        "cleaned_records": cleaned_records,
        "is_clean": not has_noise,
    }


def save_cleaned_json(output_path: Path, cleaned_records: List[Dict[str, object]]) -> None:
    output_path.write_text(
        json.dumps(cleaned_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check SQL quality for RAG ingestion")
    parser.add_argument("--sql-path", default="material_489.sql", help="Path to SQL file")
    parser.add_argument("--output", default="cleaned_material.json", help="Output cleaned JSON path")
    args = parser.parse_args()

    sql_path = Path(args.sql_path)
    if not sql_path.exists():
        raise FileNotFoundError(f"SQL file not found: {sql_path}")

    analysis = analyze_sql(sql_path)
    report = analysis["report"]

    print("=" * 72)
    print("SQL QUALITY REPORT")
    print("=" * 72)
    print(f"File: {report['sql_file']}")
    print(f"Tables: {report['table_count']} -> {report['tables']}")
    print(f"INSERT statements: {report['insert_statement_count']}")
    print(f"Total records: {report['total_records']}")
    print(f"Suspicious literals: {report['suspicious_literal_count']}")
    print()
    print("Random sample (up to 3 rows):")
    for idx, sample in enumerate(report["sample_findings"], start=1):
        print(f"[{idx}] table={sample['table']} issues={sample['issues']}")
        print(sample["preview"])
        print("-" * 72)

    print("Conclusion:")
    print(f"- Data status: {'clean' if analysis['is_clean'] else 'not clean'}")
    print(f"- RAG readiness: {report['suitability']}")
    if report["reasons"]:
        for r in report["reasons"]:
            print(f"  - {r}")

    if not analysis["is_clean"]:
        cleaned_path = Path(args.output)
        save_cleaned_json(cleaned_path, analysis["cleaned_records"])
        print()
        print("Cleaning actions applied:")
        print("- Remove HTML tags")
        print("- Remove control chars and normalize whitespace")
        print("- Filter rows with cleaned text length < 10")
        print("- Prefer text columns: question/description/answer/content/title/body/summary/remark")
        print(f"Cleaned output saved: {cleaned_path.resolve()}")
        print(f"Cleaned records: {report['cleaned_record_count']}")


if __name__ == "__main__":
    main()
