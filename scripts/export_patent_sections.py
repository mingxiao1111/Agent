from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def extract_md_sections(md_text: str) -> list[dict[str, Any]]:
    lines = md_text.splitlines()
    heads: list[tuple[int, str]] = []
    for idx, line in enumerate(lines, start=1):
        m = re.match(r"^\s*#\s+(.+?)\s*$", line)
        if not m:
            continue
        title = str(m.group(1)).strip()
        if title:
            heads.append((idx, title))

    sections: list[dict[str, Any]] = []
    for i, (line_no, title) in enumerate(heads):
        next_line = heads[i + 1][0] if i + 1 < len(heads) else len(lines) + 1
        body_lines = lines[line_no : next_line - 1]
        body = "\n".join(body_lines).strip()
        sections.append({"title": title, "line_no": line_no, "body": body})
    return sections


def extract_bracket_field(text: str, field: str, limit: int = 220) -> str:
    pattern = rf"【{re.escape(field)}】\s*(.*?)(?=【[^】]+】|$)"
    m = re.search(pattern, text, re.S)
    if not m:
        return ""
    value = re.sub(r"\s+", " ", str(m.group(1)).strip())
    return value[:limit]


def looks_like_patent_medicine_section(title: str, body: str) -> bool:
    title_text = str(title or "").strip()
    body_text = str(body or "").strip()
    if not title_text or not body_text:
        return False

    compact = re.sub(r"\s+", " ", body_text)
    if len(compact) < 60:
        return False

    markers = ["【功能与主治】", "【药物组成】", "【用法与用量】", "【禁忌】", "【注意事项】"]
    return any(marker in body_text for marker in markers)


def export_patent_sections(root: Path) -> tuple[Path, Path]:
    all_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for path in sorted(root.glob("*.md")):
        raw = ""
        try:
            raw = path.read_text(encoding="utf-8-sig")
        except Exception:
            try:
                raw = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
        if not raw.strip():
            continue

        for sec in extract_md_sections(raw):
            title = str(sec.get("title", "")).strip()
            line_no = int(sec.get("line_no", 0) or 0)
            body = str(sec.get("body", "")).strip()
            is_candidate = looks_like_patent_medicine_section(title, body)

            row = {
                "doc_id": f"{path.name}:{line_no}",
                "file": path.name,
                "line_no": line_no,
                "title": title,
                "is_candidate": is_candidate,
                "indication": extract_bracket_field(body, "功能与主治", limit=260),
                "contraindications": extract_bracket_field(body, "禁忌", limit=180),
                "cautions": extract_bracket_field(body, "注意事项", limit=220),
                "text": f"# {title}\n{body}".strip(),
            }
            all_rows.append(row)
            if is_candidate:
                candidate_rows.append(row)

    all_out = root / "patent_sections_all.jsonl"
    cand_out = root / "patent_sections_candidates.jsonl"
    with all_out.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with cand_out.open("w", encoding="utf-8") as f:
        for row in candidate_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"exported all={len(all_rows)} -> {all_out}")
    print(f"exported candidates={len(candidate_rows)} -> {cand_out}")
    return all_out, cand_out


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    root = base_dir / "data" / "中成药"
    if not root.exists():
        raise SystemExit(f"not found: {root}")
    export_patent_sections(root)


if __name__ == "__main__":
    main()
