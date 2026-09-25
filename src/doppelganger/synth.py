"""Generator inversion: labelled synthetic pairs for a country that has no labels (France).

The challenge data is produced by a noise generator whose operators we reverse-engineered (docs/RESEARCH_REPORT.md).
This module re-applies those operators to *unlabelled* Source-1 records of the target country:

  * operator RATES come from labelled pairs of the source countries (how often a true record's name is
    reordered / typo'd / aliased / turned into a URL / replaced by a pseudo-word, how often an address loses
    components, gains an injected door number, swaps its region, how far a decoy's house number moves);
  * operator VOCABULARY comes only from the target country's own unlabelled records (descriptor words and
    legal forms from its Source-1 names, pseudo-names / alias markers / number prefixes / street abbreviations
    observed in its Source-2/3 records, region variants from the learned lexicon).

Each synthetic entity yields noisy true variants plus sibling decoys (same street, shifted house number,
descriptor-word change) — the pattern that makes up the unmatched Source-2/3 records.
Validation (pipeline.loco_experiment): pretend India is unlabelled, generate synthetic India from US-only
rates, and measure India holdout F0.5 with vs. without the synthetic block.
"""
import json
import re
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import pyarrow as pa
from rapidfuzz import fuzz

from . import normalize as nz

ACCENT = str.maketrans({"e": "é", "o": "ô", "a": "à", "i": "í", "u": "ü", "c": "ç"})


@dataclass
class Profile:
    rates: dict           # operator probabilities (source countries)
    n_variants: list      # empirical matches-per-entity sample (source countries)
    decoy_delta: list     # empirical |house-number shift| of look-alike decoys (source countries)
    decoys_per_entity: float
    descriptors: list     # target-country vocabulary ...
    legal: list
    pseudo: list
    markers: list
    num_prefix: list
    region_variants: dict
    abbrev: dict

    def save(self, p):
        p.write_text(json.dumps(asdict(self), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, p):
        return cls(**json.loads(p.read_text(encoding="utf-8")))


# ------------------------------------------------------------------------------------------ measuring rates
def _toks(s):
    return re.findall(r"[^\W_]+", s.lower())


def _near_miss(a, b):
    """A misspelt address word: a candidate token absent from S1 but >= 80% similar to one of its tokens."""
    sa = set(a)
    return any(t not in sa and len(t) >= 4 and not t.isdigit() and max((fuzz.ratio(t, x) for x in sa), default=0) >= 80
               for t in b)


def measure_rates(s1_raw: pd.DataFrame, c_raw: pd.DataFrame) -> dict:
    """Operator frequencies on aligned (S1, true cand) raw record pairs."""
    n1, n2 = s1_raw.business_name.tolist(), c_raw.business_name.tolist()
    a1, a2 = s1_raw.business_address.tolist(), c_raw.business_address.tolist()
    t1, t2 = [_toks(x) for x in n1], [_toks(x) for x in n2]
    alias = c_raw.business_name.str.contains(nz.ALIAS_RE[4:].replace("(?P<pre>", "(?:").replace("(?P<real>", "(?:"),
                                             case=False, regex=True)
    url = c_raw.business_name.str.contains(r"\.(?:com|in|net|org|fr|co)\b|^[@#]|www\.", case=False, regex=True)
    junk = c_raw.business_name.str.match(r"^[^\w@#]+\s")
    single = np.array([len(b) == 1 and len(a) > 1 and b[0] not in a for a, b in zip(t1, t2)])
    pseudo = single & ~url.values & ~alias.values
    plain = ~(alias.values | url.values | pseudo)
    def rate(f):
        return float(np.mean([f(a, b) for a, b, ok in zip(t1, t2, plain) if ok]))
    c1 = [[p.strip().lower() for p in x.split(",") if p.strip()] for x in a1]
    c2 = [[p.strip().lower() for p in x.split(",") if p.strip()] for x in a2]
    d1 = [re.findall(r"\d+", x) for x in a1]
    d2 = [re.findall(r"\d+", x) for x in a2]
    has = [bool(x) for x in c2]
    upper = lambda s: s.str.upper().eq(s) & s.str.contains(r"[A-Za-z]")
    return {
        "pseudo": float(pseudo.mean()), "alias": float(alias.mean()), "url": float(url.mean()),
        "junk": float(junk.mean()), "upper_name": float(upper(c_raw.business_name).mean()),
        "reorder": rate(lambda a, b: sorted(a) == sorted(b) and a != b),
        "drop": rate(lambda a, b: len(b) < len(a) and set(b) < set(a)),
        "add": rate(lambda a, b: len(b) > len(a) and set(a) < set(b)),
        "typo": rate(lambda a, b: len(a) == len(b) and sorted(a) != sorted(b) and fuzz.ratio(" ".join(a), " ".join(b)) >= 75),
        "accent": float(c_raw.business_name.str.contains(r"[À-ÿ]").mean()),
        "empty_addr": float(1 - np.mean(has)),
        "upper_addr": float(upper(c_raw.business_address[np.array(has)]).mean()),
        "comp_reorder": float(np.mean([sorted(a) == sorted(b) and a != b for a, b, h in zip(c1, c2, has) if h])),
        "comp_drop": float(np.mean([len(b) < len(a) for a, b, h in zip(c1, c2, has) if h])),
        "num_missing": float(np.mean([bool(x) and not y for x, y, h in zip(d1, d2, has) if h])),
        "num_inject": float(np.mean([bool(set(y) - set(x)) for x, y, h in zip(d1, d2, has) if h and x])),
        "addr_typo": float(np.mean([_near_miss(_toks(a), _toks(b)) for a, b, h in zip(a1, a2, has) if h])),
    }


def target_vocabulary(s1_raw: pd.DataFrame, c_raw: pd.DataFrame, c_pseudo: np.ndarray, lex, country) -> dict:
    """Everything country-specific, learned from the target country's unlabelled records only."""
    tok = pd.Series([t for x in s1_raw.business_name for t in _toks(x)]).value_counts()
    legal = [t for t in tok.index[:400] if t in nz.LEGAL][:20]
    desc = [t for t in tok.index[:300] if t not in nz.LEGAL and t.isalpha() and len(t) > 2][:150]
    names = c_raw.business_name
    mk = names.str.extract(nz.ALIAS_RE[4:].replace("(?:", "(?P<m>", 1), flags=re.I)["m"].dropna().str.strip()
    markers = mk.value_counts().head(8).index.tolist() or ["DBA"]
    pref = c_raw.business_address.str.extract(r"(?:^|,\s*)([^\w\s,]{1,2}\s?|[A-Za-zº°\.]{1,5}\s?[º°#]?\s?)\d", expand=False)
    num_prefix = pref.dropna().str.strip().value_counts().head(8).index.tolist()
    inv = {}
    for k, v in lex.components.items():
        c, variant = k.split("|", 1)
        if c == country:
            inv.setdefault(v, []).append(variant)
    words = pd.Series([t for x in c_raw.business_address for t in re.findall(r"[A-Za-zÀ-ÿ\.]+", x)]).value_counts()
    abbrev = {}
    for surface, cnt in words.head(3000).items():
        canon = nz.ABBREV.get(surface.lower().rstrip("."))
        if canon and cnt >= 50:
            abbrev.setdefault(canon, []).append(surface)
    return dict(descriptors=desc, legal=legal, pseudo=names[c_pseudo].head(5000).tolist(), markers=markers,
                num_prefix=num_prefix, region_variants=inv, abbrev=abbrev)


# ------------------------------------------------------------------------------------------ generation
class Generator:
    def __init__(self, prof: Profile, seed: int):
        self.p, self.r = prof, np.random.default_rng(seed)

    def _chance(self, k, scale=1.0):
        return self.r.random() < self.p.rates[k] * scale

    def _pick(self, seq, default=""):
        return seq[self.r.integers(len(seq))] if len(seq) else default

    def _typo(self, w):
        if len(w) < 4:
            return w
        i, op = self.r.integers(1, len(w) - 1), self.r.integers(4)
        return [w[:i] + w[i + 1] + w[i] + w[i + 2:], w[:i] + w[i + 1:], w[:i] + w[i] + w[i:],
                w[:i] + chr(97 + self.r.integers(26)) + w[i + 1:]][op]

    def name(self, name: str, decoy=False) -> str:
        p = self.p
        w = name.split()
        if decoy:   # sibling business: descriptor word added or swapped
            d = self._pick(p.descriptors).title()
            w = w + [d] if self.r.random() < 0.6 or len(w) < 2 else w[:-1] + [d]
        elif self._chance("pseudo"):
            return self._pick(p.pseudo, "Zorvex")
        if self._chance("reorder") and len(w) > 1:
            i = self.r.integers(len(w) - 1)
            w[i], w[i + 1] = w[i + 1], w[i]
        if self._chance("drop") and len(w) > 1:
            del w[self.r.integers(1, len(w))]
        if self._chance("add"):
            w.append(self._pick(p.legal).upper() if self.r.random() < 0.5 else self._pick(p.descriptors).title())
        if self._chance("typo"):
            i = self.r.integers(len(w))
            w[i] = self._typo(w[i])
        if self._chance("accent"):
            i = self.r.integers(len(w))
            w[i] = w[i].translate(ACCENT) if self.r.random() < 0.3 else w[i][:1] + w[i][1:].translate(ACCENT)
        s = " ".join(w)
        if not decoy and self._chance("url"):
            s = re.sub(r"[^a-z0-9]", "", s.lower()) + ".fr" if self.r.random() < 0.7 else "@" + re.sub(r"[^a-z0-9]", "", s.lower())
        if not decoy and self._chance("alias"):
            s = f"{self._pick(p.pseudo, 'Zorvex')} {self._pick(p.markers, 'DBA')} {s}"
        if self._chance("upper_name"):
            s = s.upper()
        if self._chance("junk"):
            s = self._pick([">> ", "-- ", "<< ", "*** ", "... "]) + s
        return s

    def address(self, addr: str, decoy=False) -> str:
        p = self.p
        comps = [c.strip() for c in addr.split(",") if c.strip()]
        if decoy:    # same street, house number shifted by an empirically distributed delta
            for i, c in enumerate(comps):
                m = re.search(r"\d+", c)
                if m:
                    n = int(m.group()) + int(self._pick(p.decoy_delta, 7)) * (1 if self.r.random() < 0.5 else -1)
                    comps[i] = c[:m.start()] + str(max(n, 1)) + c[m.end():]
                    break
        elif self._chance("empty_addr"):
            return ""
        comps = [self._pick(p.region_variants.get(nz_key(c), [c])) if nz_key(c) in p.region_variants
                 and self.r.random() < 0.5 else c for c in comps]
        if self._chance("comp_drop") and len(comps) > 2:
            del comps[self.r.integers(1, len(comps))]
        if self._chance("comp_reorder") and len(comps) > 1:
            comps = [comps[i] for i in self.r.permutation(len(comps))]
        s = ", ".join(comps)
        for canon, surf in p.abbrev.items():
            s = re.sub(rf"\b({'|'.join(map(re.escape, surf))})\b", lambda m: self._pick(surf), s) if self.r.random() < 0.5 else s
        if self._chance("num_missing"):
            s = re.sub(r"^[\s,]*\d+\w*[\s,]*", "", s, count=1)
        elif self._chance("num_inject"):
            s = f"{self._pick(p.num_prefix, 'N°')} {self.r.integers(1, 400)} {s}"
        if self._chance("addr_typo"):
            w = s.split(" ")
            i = self.r.integers(len(w))
            w[i] = self._typo(w[i]) if w[i].isalpha() else w[i]
            s = " ".join(w)
        return s.upper() if self._chance("upper_addr") else s

    def generate(self, s1_raw: pd.DataFrame, max_entities: int, prefix="SYN"):
        """Raw synthetic records + origin S1 row (label) + decoy flag."""
        ents = s1_raw.sample(min(max_entities, len(s1_raw)), random_state=int(self.r.integers(1 << 30)))
        rows = []
        for row, r in zip(ents.index, ents.itertuples()):
            for _ in range(int(self._pick(self.p.n_variants, 3))):
                rows.append((self.name(r.business_name), self.address(r.business_address), r.country, row, False))
            for _ in range(self.r.poisson(self.p.decoys_per_entity)):
                rows.append((self.name(r.business_name, True), self.address(r.business_address, True), r.country, row, True))
        df = pd.DataFrame(rows, columns=["business_name", "business_address", "country", "origin", "decoy"])
        df.insert(0, "entity_id", [f"S{2 + (i % 2)}-{prefix}{i}" for i in range(len(df))])
        return df


def nz_key(c):
    return re.sub(r"[^\w]+", " ", c.lower()).strip()


def to_table(df: pd.DataFrame) -> pa.Table:
    return pa.Table.from_pandas(df[["entity_id", "business_name", "business_address", "country"]], preserve_index=False)
