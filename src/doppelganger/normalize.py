"""Vectorised (pyarrow.compute) normalisation of business names and addresses.

Two passes:
  base_normalize()  -- lexicon-free cleaning: folds case/accents, detects noise operators
                       (alias marker, URL/handle, junk prefix, native script), splits names into
                       tokens and addresses into comma components, extracts address numbers.
  finalize()        -- applies the learned Lexicon (transliteration, region/locality equivalences)
                       and derives the strings/tokens the blocker and feature extractor consume.
"""
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

# Alias/rebrand markers observed in the data: "<pseudo> formerly known as <real>", "X t/a Y", "X DBA Y" ...
ALIAS_RE = (r"(?i)^(?P<pre>.*?)\s+(?:formerly known as|formerly:?|f/k/a|fka|a/k/a|aka|t/a|d/b/a|dba:?|"
            r"trading as)\s+(?P<real>.+)$")
URL_RE = r"(?:\.(?:com|in|net|org|fr|co|io)\b|^[@#]|www\.)"
SCRIPT_RE = r"[\x{0370}-\x{FFFF}]"          # any non-Latin letter (Devanagari, Tamil, Bengali, ...)
LEGAL = set("""llc inc incorporated corp corporation co company ltd limited pvt private llp lp pllc pc plc pa
the of and et ms sri shri smt dr sarl sas sasu eurl sa sci ei cie snc selarl ets""".split())
# Street-type / direction abbreviations -> one canonical short form (both sides get the same form).
ABBREV = {**dict.fromkeys(["street", "saint", "st", "str"], "st"), **dict.fromkeys(["road", "rd"], "rd"),
          **dict.fromkeys(["drive", "dr"], "dr"), **dict.fromkeys(["avenue", "ave", "av"], "ave"),
          **dict.fromkeys(["boulevard", "blvd", "bd", "boul"], "blvd"), **dict.fromkeys(["lane", "ln"], "ln"),
          **dict.fromkeys(["court", "ct"], "ct"), **dict.fromkeys(["place", "pl"], "pl"),
          **dict.fromkeys(["rue", "r"], "rue"), **dict.fromkeys(["impasse", "imp"], "imp"),
          **dict.fromkeys(["chemin", "che", "chem"], "chemin"), **dict.fromkeys(["trail", "trl"], "trl"),
          **dict.fromkeys(["highway", "hwy"], "hwy"), **dict.fromkeys(["circle", "cir"], "cir"),
          **dict.fromkeys(["parkway", "pkwy"], "pkwy"), **dict.fromkeys(["north", "n"], "n"),
          **dict.fromkeys(["south", "s"], "s"), **dict.fromkeys(["east", "e"], "e"),
          **dict.fromkeys(["west", "w"], "w"), **dict.fromkeys(["sainte", "ste"], "ste"),
          "apartment": "apt", "suite": "ste", "building": "bldg", "floor": "flr", "nagar": "ngr"}
# Tokens injected as noise around numbers / placeholders ("H.no 954", "Door No", "N°", "null", "CDP").
ADDR_STOP = {"no", "nº", "hno", "h", "door", "null", "n/a", "na", "cdp", "fcdp", "city", "of", "the", "unit"}


def _re(a, pat, rep=" "):
    return pc.replace_substring_regex(a, pat, rep)


def _squeeze(a):
    return pc.utf8_trim_whitespace(_re(a, r"\s+", " "))


def fold(a):
    """lowercase + strip Latin diacritics only (NFKD then drop U+0300-036F, keeping Indic vowel signs)."""
    a = pc.utf8_normalize(pc.utf8_lower(pc.fill_null(a, "")), "NFKD")
    return pc.utf8_normalize(_re(a, r"[\x{0300}-\x{036f}]", ""), "NFC")


def split_tokens(s, sep=" "):
    """Split strings into a ListArray dropping empty pieces."""
    s = pa.chunked_array([s]).combine_chunks() if isinstance(s, pa.Array) else s.combine_chunks()
    lst = pc.split_pattern(s, sep)
    vals, par = pc.list_flatten(lst), pc.list_parent_indices(lst)
    keep = pc.not_equal(pc.utf8_trim_whitespace(vals), "")
    vals, par = pc.utf8_trim_whitespace(vals.filter(keep)), par.filter(keep).to_numpy()
    return _from_parents(vals, par, len(s))


def _from_parents(vals, parents, n):
    offsets = np.zeros(n + 1, np.int32)
    np.cumsum(np.bincount(parents, minlength=n), out=offsets[1:])
    return pa.ListArray.from_arrays(pa.array(offsets), vals)


def map_values(values, mapping):
    """Vectorised dict lookup; unmapped values pass through."""
    if not mapping:
        return values
    idx = pc.index_in(values, value_set=pa.array(list(mapping)))
    return pc.if_else(pc.is_valid(idx), pc.take(pa.array(list(mapping.values())), idx), values)


def map_list(lst, fn):
    """Apply a value-level function to every element of a ListArray (keeps the list structure)."""
    return pa.ListArray.from_arrays(lst.offsets, fn(lst.flatten()))


def filter_list(lst, mask_fn):
    vals, par = lst.flatten(), pc.list_parent_indices(lst)
    keep = mask_fn(vals)
    return _from_parents(vals.filter(keep), par.filter(keep).to_numpy(), len(lst))


def number_matrix(addr_folded, k):
    """First k digit-runs of each address as int32 (leading zeros stripped, -1 = absent)."""
    digits = split_tokens(_squeeze(_re(addr_folded, r"\D+", " ")))
    vals = pc.utf8_slice_codeunits(_re(digits.flatten(), r"^0+(\d)", r"\1"), -9)
    vals = pc.cast(vals, pa.int32()).to_numpy()
    off, n = digits.offsets.to_numpy(), len(digits)
    lens = np.diff(off)
    m = np.full((n, k), -1, np.int32)
    for j in range(k):
        has = lens > j
        m[has, j] = vals[off[:-1][has] + j]
    return m, lens.astype(np.int16)


def base_normalize(t: pa.Table, max_numbers: int) -> pa.Table:
    """Lexicon-free normalisation of a raw source table (entity_id, business_name, business_address, country)."""
    t = t.cast(pa.schema([(c, pa.string()) for c in ("entity_id", "business_name", "business_address", "country")]))
    raw = pc.fill_null(t["business_name"], "").combine_chunks()
    al = pc.extract_regex(raw, ALIAS_RE)
    f_alias = pc.and_(pc.is_valid(al), pc.not_equal(pc.fill_null(pc.struct_field(al, "real"), ""), ""))
    name = pc.if_else(f_alias, pc.struct_field(al, "real"), raw)
    name = fold(name)
    name = _re(name, r"\(\s*id:?\s*\d+\s*\)|\s*\|.*$|\bm/s\b", " ")      # "(ID: 123)", "| www.x.com", "M/s"
    f_junk = pc.match_substring_regex(name, r"^[^\p{L}\p{M}\p{N}@#]+")
    name = _re(name, r"^[^\p{L}\p{M}\p{N}@#]+", "")
    f_url = pc.match_substring_regex(name, URL_RE)
    name = _re(name, r"www\.|\.(?:com|in|net|org|fr|co|io)\b|^[@#]", " ")
    name = _re(name, r"\b(\w)\.", r"\1")                                   # L.L.C. -> llc, S.A.S. -> sas
    name = _re(name, "&", " and ")
    name = _squeeze(_re(name, r"[^\p{L}\p{M}\p{N}]+", " "))

    addr = fold(t["business_address"].combine_chunks())
    addr = _re(addr, r"\bp\.?\s?o\.?\s*box\s*\d+|\bnull\b|\bn/a\b", " ")     # injected PO boxes / placeholders
    nums, n_nums = number_matrix(addr, max_numbers)
    comps = split_tokens(addr, ",")
    comps = filter_list(map_list(comps, lambda v: _squeeze(_re(v, r"[^\p{L}\p{M}\p{N}]+", " "))),
                        lambda v: pc.not_equal(v, ""))
    cols = {
        "entity_id": t["entity_id"].combine_chunks(), "country": t["country"].combine_chunks(),
        "name_tokens": split_tokens(name), "addr_comps": comps,
        "f_alias": f_alias, "f_url": f_url, "f_junk": f_junk,
        "f_script": pc.match_substring_regex(name, SCRIPT_RE),
        "f_empty_addr": pc.equal(pc.list_value_length(comps), 0),
        "n_nums": pa.array(n_nums),
    }
    cols.update({f"num{j}": pa.array(nums[:, j]) for j in range(max_numbers)})
    return pa.table(cols)


def core_name(tok):
    """(full, core) strings from name tokens; core drops legal/generic tokens unless nothing would remain."""
    full = pc.binary_join(tok, " ")
    core = pc.binary_join(filter_list(tok, lambda v: pc.invert(pc.is_in(v, value_set=pa.array(sorted(LEGAL))))), " ")
    return full, pc.if_else(pc.equal(core, ""), full, core)


HOMOGLYPHS = {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "8": "b"}


def fold_homoglyphs(v: pa.Array) -> pa.Array:
    """Digits inside tokens that also contain letters are read as letters ("c0mmerce" -> "commerce")."""
    mixed = pc.and_(pc.match_substring_regex(v, r"\p{L}"), pc.match_substring_regex(v, r"\d"))
    f = v
    for d, ch in HOMOGLYPHS.items():
        f = pc.replace_substring(f, d, ch)
    return pc.if_else(mixed, f, v)


def segment(word: str, vocab: set, max_len: int = 24) -> tuple[list, int]:
    """Split a space-free name into vocabulary words (fewest pieces; an unknown character costs 1.5).
    Returns the pieces and the number of characters they cover."""
    n = len(word)
    best, back = [0.0] + [1e9] * n, [None] * (n + 1)
    for j in range(1, n + 1):
        c, b = best[j - 1] + 1.5, (j - 1, False)
        for i in range(max(0, j - max_len), j - 1):
            if best[i] + 1 < c and word[i:j] in vocab:
                c, b = best[i] + 1, (i, True)
        best[j], back[j] = c, b
    out, j = [], n
    while j > 0:
        i, known = back[j]
        if known:
            out.append(word[i:j])
        j = i
    return out[::-1], sum(map(len, out))


def segment_urls(tok: pa.ListArray, nospace: pa.Array, f_url: pa.Array, vocab: set, min_cover=0.6) -> pa.ListArray:
    """URL/handle names ("impexprivate.com") -> vocabulary words, when they explain most of the string."""
    rows = np.flatnonzero(f_url.to_numpy(zero_copy_only=False))
    if not len(rows):
        return tok
    py, ns = tok.to_pylist(), nospace.take(pa.array(rows)).to_pylist()
    for i, w in zip(rows, ns):
        if w and len(w) >= 4:
            pieces, cover = segment(w, vocab)
            if len(pieces) > 1 and cover >= min_cover * len(w):
                py[i] = pieces
    return pa.array(py, pa.list_(pa.string()))


def finalize(t: pa.Table, lex) -> pa.Table:
    """Apply learned lexicon and derive matching representations."""
    ctry = t["country"].combine_chunks()
    # names: transliterate native-script tokens, split legal/generic tokens from the core
    tok = map_list(t["name_tokens"].combine_chunks(), lambda v: map_values(v, lex.translit))
    tok = map_list(tok, fold_homoglyphs)
    full, core = core_name(tok)
    nospace = _re(core, " ", "")
    tok = segment_urls(tok, nospace, t["f_url"].combine_chunks(), lex.vocab_set())
    full, core = core_name(tok)
    core_l = split_tokens(core)
    n_core = pc.list_value_length(core_l)
    # pseudo-word name: a single out-of-vocabulary Latin token that is not a URL/handle
    single = pc.and_(pc.equal(n_core, 1), pc.invert(t["f_url"].combine_chunks()))
    f_pseudo = pc.and_(pc.and_(single, pc.invert(pc.is_in(core, value_set=lex.vocab_array()))),
                       pc.and_(pc.greater_equal(pc.utf8_length(core), lex.pseudo_min_len),
                               pc.match_substring_regex(core, r"^[a-z]+$")))
    # addresses: country-scoped component equivalences (state codes, native-script states, départements)
    comps = t["addr_comps"].combine_chunks()
    par = pc.list_parent_indices(comps)
    keyed = pc.binary_join_element_wise(pc.take(ctry, par), comps.flatten(), "|")
    mapped = pc.take(pa.array(list(lex.components.values()) or [""]), pc.index_in(keyed, value_set=pa.array(list(lex.components) or ["\x00"])))
    comp_vals = pc.if_else(pc.is_valid(mapped), mapped, comps.flatten())
    comps = pa.ListArray.from_arrays(comps.offsets, comp_vals)
    loc_keyed = pc.binary_join_element_wise(pc.take(ctry, par), comp_vals, "|")
    loc = filter_list(pa.ListArray.from_arrays(comps.offsets, loc_keyed),
                      lambda v: pc.is_in(v, value_set=lex.locality_array()))
    atok = map_list(split_tokens(pc.binary_join(comps, " ")), lambda v: map_values(v, ABBREV))
    atok = filter_list(atok, lambda v: pc.invert(pc.is_in(v, value_set=pa.array(sorted(ADDR_STOP)))))
    alpha = filter_list(atok, lambda v: pc.invert(pc.match_substring_regex(v, r"\d")))
    out = {c: t[c] for c in t.column_names if c not in ("name_tokens", "addr_comps")}
    out.update(name_full=full, name_core=core, name_nospace=nospace, name_tokens=core_l,
               n_name_tok=n_core.cast(pa.int16()), f_pseudo=f_pseudo,
               addr_norm=pc.binary_join(atok, " "), addr_alpha=pc.binary_join(alpha, " "),
               addr_tokens=alpha, localities=loc)
    return pa.table(out)
