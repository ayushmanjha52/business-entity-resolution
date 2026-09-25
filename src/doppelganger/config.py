"""Single configuration layer: TOML file -> nested attribute namespace."""
import tomllib
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]


def _ns(d):
    return SimpleNamespace(**{k: _ns(v) if isinstance(v, dict) else v for k, v in d.items()})


def load(path: str | Path = ROOT / "configs" / "default.toml") -> SimpleNamespace:
    cfg = _ns(tomllib.loads(Path(path).read_text(encoding="utf-8")))
    for k in ("data_dir", "work_dir", "output_dir"):  # resolve relative to repo root
        p = Path(getattr(cfg.paths, k))
        setattr(cfg.paths, k, p if p.is_absolute() else ROOT / p)
    cfg.paths.work_dir.mkdir(parents=True, exist_ok=True)
    return cfg
