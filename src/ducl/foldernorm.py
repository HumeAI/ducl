"""Collapse high-cardinality folder names so they group informatively.

A normalization rule is a single **regex**.  Within each match, the spans
matched by **capture groups are collapsed to ``*``**; everything else (literal
text, and any non-captured part of the match) is **preserved**.  Rules are
applied to each path *component* in order.

Examples (regex -> effect):
    ``(\\d{7,})``          collapse 7+ digit runs (timestamps/epochs/shards)
                           delivery-20260423        -> delivery-*
                           room-RM09a04e-1779919627 -> room-RM09a04e-*
    ``delivery-(.*)``      fold every delivery (also resharded-delivery-...)
                           resharded-delivery-2026-03-05-x -> resharded-delivery-*
    ``.+-batch-(.*)``      keep the prefix, fold the number
                           zh-batch-0 -> zh-batch-*

A regex with no capture groups collapses nothing (everything is preserved), so
remember to wrap the varying part in ``(...)``.

The collapse is computed in Python over the *unique* components in the data and
then applied vectorially via a Polars mapping -- so it works component-by-
component on whole path columns and on standalone component columns
(``leaf_folder``), and component counts are unchanged (matches never cross
``/``), so callers needn't recompute ``n_components``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import polars as pl

# Convenience rule for the "digits longer than 6 chars" heuristic.
DIGIT_RULE = r"(\d{7,})"


def _collapse_one(patterns: list[re.Pattern], comp: str) -> str:
    """Apply each compiled rule to a single component, ``*``-ing captured spans."""
    for pat in patterns:
        def repl(m: re.Match) -> str:
            text = m.group(0)
            base = m.start()
            spans = [
                (m.start(i) - base, m.end(i) - base)
                for i in range(1, m.re.groups + 1)
                if m.group(i) is not None
            ]
            for a, b in sorted(spans, reverse=True):  # right-to-left keeps offsets valid
                text = text[:a] + "*" + text[b:]
            return text
        comp = pat.sub(repl, comp)
    return comp


def build_mapping(components: Iterable[str], rules: list[str]) -> dict[str, str]:
    """Map each component that changes -> its collapsed form."""
    patterns = [re.compile(r) for r in rules]
    mapping: dict[str, str] = {}
    for comp in components:
        if comp is None:
            continue
        collapsed = _collapse_one(patterns, comp)
        if collapsed != comp:
            mapping[comp] = collapsed
    return mapping


def _unique_path_components(df: pl.DataFrame, col: str) -> set[str]:
    s = (
        df.select(pl.col(col).str.split("/").alias("_c"))
        .explode("_c")["_c"].drop_nulls().unique()
    )
    return set(s.to_list())


def normalize_columns(
    df: pl.DataFrame,
    rules: list[str] | None,
    *,
    path_cols: tuple[str, ...] = (),
    component_cols: tuple[str, ...] = (),
) -> pl.DataFrame:
    """Collapse folder names in *df*.

    ``path_cols`` hold full ``/``-separated paths (normalized component by
    component); ``component_cols`` hold single components (e.g. ``leaf_folder``).
    Rows are not merged here -- downstream group-bys fold the now-identical
    names together.
    """
    if not rules:
        return df

    comps: set[str] = set()
    for c in path_cols:
        if c in df.columns:
            comps |= _unique_path_components(df, c)
    for c in component_cols:
        if c in df.columns:
            comps |= set(df[c].drop_nulls().unique().to_list())

    mapping = build_mapping(comps, rules)
    if not mapping:
        return df

    exprs: list[pl.Expr] = []
    for c in path_cols:
        if c in df.columns:
            exprs.append(
                pl.col(c).str.split("/")
                .list.eval(pl.element().replace(mapping))
                .list.join("/").alias(c)
            )
    for c in component_cols:
        if c in df.columns:
            exprs.append(pl.col(c).replace(mapping).alias(c))
    return df.with_columns(exprs)
