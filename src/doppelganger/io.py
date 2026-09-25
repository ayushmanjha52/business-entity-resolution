"""Data access: raw TSV -> cached normalised parquet, ground truth, and submission writers."""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

from . import normalize as nz
from .utils import log

SOURCES = (1, 2, 3)


def raw_path(cfg, split, src):
    return cfg.paths.data_dir / split / f"{split}_source{src}.tsv"


def read_raw(path: Path) -> pa.Table:
    """Read a challenge TSV verbatim (tabs only, no quoting, every column a string, '' kept)."""
    names = ["entity_id", "business_name", "business_address", "country"]
    return pcsv.read_csv(path, parse_options=pcsv.ParseOptions(delimiter="\t", quote_char=False),
                         convert_options=pcsv.ConvertOptions(column_types={c: pa.string() for c in names},
                                                             strings_can_be_null=False))


def _cached(path: Path, build):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        build(tmp)
        tmp.replace(path)
    return pq.read_table(path)


def _chunked(table, fn, out, chunk):
    w = None
    for i in range(0, table.num_rows, chunk):
        part = fn(table.slice(i, chunk))
        w = w or pq.ParquetWriter(out, part.schema)
        w.write_table(part)
    w.close()


def base_table(cfg, split, src) -> pa.Table:
    p = cfg.paths.work_dir / "base" / f"{split}_s{src}.parquet"
    return _cached(p, lambda out: _chunked(read_raw(raw_path(cfg, split, src)),
                                           lambda t: nz.base_normalize(t, cfg.prep.max_numbers),
                                           out, cfg.blocking.chunk_size))


def final_table(cfg, split, src, lex) -> pa.Table:
    p = cfg.paths.work_dir / "final" / f"{split}_s{src}.parquet"
    return _cached(p, lambda out: _chunked(base_table(cfg, split, src), lambda t: nz.finalize(t, lex),
                                           out, cfg.blocking.chunk_size))


def load_split(cfg, split, lex=None, columns=None):
    """(s1, cand) tables; cand = Source 2 rows followed by Source 3 rows (row index = global cand id)."""
    get = (lambda s: final_table(cfg, split, s, lex)) if lex is not None else (lambda s: base_table(cfg, split, s))
    s1 = get(1)
    cand = pa.concat_tables([get(2), get(3)])
    if columns:
        s1, cand = s1.select(columns), cand.select(columns)
    log(f"{split}: S1={s1.num_rows:,} S2+S3={cand.num_rows:,}")
    return s1, cand.combine_chunks()


def ground_truth(cfg, s1_ids: pa.Array, cand_ids: pa.Array):
    """Ground truth as (s1_row, cand_row) int arrays aligned to the loaded tables."""
    gt = read_raw_gt(cfg.paths.data_dir / "train" / "train_ground_truth.tsv")
    ex = gt.assign(m=gt.matched_entity_ids.str.split(",")).explode("m")
    ex = ex[ex.m.notna() & (ex.m != "")]
    a = pd.Index(s1_ids.to_numpy(zero_copy_only=False)).get_indexer(ex.source1_entity_id)
    b = pd.Index(cand_ids.to_numpy(zero_copy_only=False)).get_indexer(ex.m)
    assert (a >= 0).all() and (b >= 0).all(), "ground truth references unknown ids"
    return a.astype(np.int32), b.astype(np.int32)


def read_raw_gt(path):
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, quoting=3)


def write_lists(path: Path, s1_ids, s1_rows, cand_rows, cand_ids, header):
    """One row per Source-1 entity; comma-joined cand ids (empty when none). Rows must be de-duplicated."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"s": s1_rows, "c": np.asarray(cand_ids)[cand_rows]})
    lists = df.groupby("s").c.agg(",".join)
    out = pd.DataFrame({header[0]: np.asarray(s1_ids)})
    out[header[1]] = lists.reindex(np.arange(len(out))).fillna("").values
    out.to_csv(path, sep="\t", index=False, lineterminator="\n")
    log(f"wrote {path} ({(out[header[1]] != '').sum():,} non-empty rows)")
