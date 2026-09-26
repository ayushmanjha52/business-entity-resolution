"""CLI:  python -m doppelganger <command> [--config configs/default.toml]

  prepare   normalise all sources + learn the lexicon (TRAIN-role labels only)   (artifacts/base, artifacts/final, artifacts/lexicon.json)
  block     candidate generation + recall report                                (artifacts/pairs_*.parquet)
  train     development stage-1/stage-2 models on TRAIN entities                 (artifacts/models)
  validate  freeze the decision floor on VALID, then report VALID               (artifacts/decision.json, artifacts/validation)
  test      one-off evaluation of TEST entities with the frozen design          (artifacts/validation/*_test*)
  final     refit on every labelled entity (+ synthetic block for France)       (artifacts/models_final*)
  predict   test inference -> output/matching_results.tsv + candidate_pairs.tsv
  loco      leave-India-out test of the synthetic-data approach
  evidence  build the explorer's evidence store
  explore   launch the explorer (http://127.0.0.1:8765)
  all       prepare -> block -> train -> validate -> test -> final -> predict (with a timing/memory benchmark)
"""
import argparse
import json
import time

from . import config, pipeline
from .utils import PEAK, TIMINGS, log, rss_gb


def _final(cfg):
    """Labelled countries: model on real labels. Unlabelled countries (France): + synthetic block, then optionally
    a self-training round on their real test pairs pseudo-labelled by the first-round model."""
    pipeline.train(cfg, tag=pipeline.FINAL, roles=pipeline.LABELLED_ROLES)
    if not cfg.synth.enabled:
        return
    from . import selfsup
    tag = pipeline.FINAL + pipeline.UNLABELLED
    extra = selfsup.france_block(cfg)
    if extra is None:
        return
    # the unlabelled-country model learns mostly from its synthetic / pseudo-labelled blocks; real labelled pairs
    # of TRAIN entities are enough alongside them and keep the refit within 16 GB
    pipeline.train(cfg, extra=extra, tag=tag)
    if cfg.synth.self_training:
        del extra
        p2 = pipeline.score(cfg, "test")[-1]
        pipeline.train(cfg, extra=selfsup.france_block(cfg, p2), tag=tag)


def _validate(cfg):
    pipeline.tune_decision(cfg)
    pipeline.evaluate(cfg, pipeline.VALID)


STEPS = {
    "prepare": lambda cfg: (pipeline.load(cfg, "train"), pipeline.load(cfg, "test")),
    "block": lambda cfg: (pipeline.blocking_report(cfg), pipeline.candidates(cfg, "test")),
    "train": lambda cfg: pipeline.train(cfg),
    "validate": _validate,
    "test": lambda cfg: pipeline.evaluate(cfg, pipeline.TEST),
    "final": _final,
    "predict": lambda cfg: pipeline.predict(cfg),
    "loco": lambda cfg: __import__("doppelganger.selfsup", fromlist=["x"]).loco_experiment(cfg),
    "evidence": lambda cfg: __import__("doppelganger.evidence", fromlist=["x"]).build(cfg),
}
ALL = ["prepare", "block", "train", "validate", "test", "final", "predict"]


def main():
    ap = argparse.ArgumentParser(prog="doppelganger", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=[*STEPS, "explore", "all"])
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = config.load(a.config) if a.config else config.load()
    t0 = time.perf_counter()
    if a.command == "explore":
        import uvicorn
        from .explorer.app import create_app
        log(f"explorer on http://{cfg.explorer.host}:{cfg.explorer.port}")
        uvicorn.run(create_app(cfg), host=cfg.explorer.host, port=cfg.explorer.port, log_level="warning")
        return
    for name in ALL if a.command == "all" else [a.command]:
        STEPS[name](cfg)
    rss_gb()
    bench_path = cfg.paths.work_dir / "benchmark.json"
    bench = json.loads(bench_path.read_text()) if bench_path.exists() else {}
    bench[a.command] = {"wall_s": round(time.perf_counter() - t0, 1), "peak_rss_gb": round(PEAK["rss_gb"], 2),
                        "stages_s": {k: round(v, 1) for k, v in TIMINGS.items()}}
    bench_path.write_text(json.dumps(bench, indent=2))
    log(f"done: {a.command} in {bench[a.command]['wall_s']}s, peak RSS {bench[a.command]['peak_rss_gb']} GB")


if __name__ == "__main__":
    main()
