"""Assemble <team>_submission.zip in the layout required by the challenge.

    python scripts/create_submission.py --team <team_name>

Includes only: output/ (2 TSVs), Documentation_template.md, and code/business_entity_resolution/
(src, configs, tests, scripts, docs, README.md, requirements.txt, pytest.ini).
Never includes: .venv, student_resource (raw data), artifacts/ (caches, models, features), __pycache__.
"""
import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP = {"__pycache__", ".pytest_cache", ".venv", "artifacts", "student_resource"}
CODE = ["src", "configs", "tests", "scripts", "docs", "README.md", "requirements.txt", "pytest.ini"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    a = ap.parse_args()
    out = ROOT / f"{a.team}_submission.zip"
    need = [ROOT / "output" / "matching_results.tsv", ROOT / "output" / "candidate_pairs.tsv",
            ROOT / "Documentation_template.md"]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        raise SystemExit(f"missing: {missing}")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in need[:2]:
            z.write(p, f"output/{p.name}")
        z.write(need[2], "Documentation_template.md")
        for item in CODE:
            src = ROOT / item
            files = [src] if src.is_file() else [f for f in src.rglob("*") if f.is_file() and not (set(f.parts) & SKIP)
                                                 and f.suffix not in (".pyc", ".parquet", ".zip")]
            for f in files:
                z.write(f, f"code/business_entity_resolution/{f.relative_to(ROOT).as_posix()}")
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
    bad = [n for n in names if set(Path(n).parts) & SKIP]
    assert not bad, f"forbidden paths in zip: {bad[:5]}"
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(names)} files)")


if __name__ == "__main__":
    main()
