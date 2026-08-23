# -*- coding: utf-8 -*-
"""
Utility helpers for Generic + PartName normalization and grouping.

This module provides functions to sanitize raw PartNames into a canonical format 
(norm_partname) and create composite keys for grouping data by product variants.
"""

import re
from typing import Optional


def _get_first_token(partname: str) -> str:
    """
    Extract the first non-empty token from a potentially multi-line or comma-separated PartName string.
    
    Args:
        partname: The raw PartName string.
        
    Returns:
        The first identified token, or an empty string.
    """
    if not partname:
        return ""
    # Split by common delimiters (newline, comma)
    for tok in re.split(r"[\n,]+", str(partname)):
        tok = tok.strip()
        if tok:
            return tok
    return ""


# --------------------------- Regex rules (constants) ---------------------------

# (Rule B1) Prefix normalization: e.g., "6CL..." or "6HL..." -> "6L..."
_PREFIX_NORM_RE = re.compile(r"^(?P<num>\d+)(?P<mid>[CH])L(?P<rest>.*)$")

# (Rule B2) Drop trailing -D or -L (also allows "_" delimiter)
# Often used for package or temperature grade suffixes that don't change the base features.
_TRAILING_D_L_RE = re.compile(r"([-_])(D|L)$", flags=re.IGNORECASE)

# (Rule B3) Drop hyphen before trailing alpha package chunk (e.g., "6L2631-XV" -> "6L2631XV")
# Note: This specific rule was disabled in the previous version to keep hyphens.
_HYPHEN_BEFORE_ALPHA_END_RE = re.compile(r"(?<=\d)-(?=[A-Z]{1,4}(?:$|[-_]))")


# --------------------------- Targeted alias overrides --------------------------

# Canonical key for known equivalents (e.g., different part numbers that are functionally identical)
_CANON_EQUIV_KEY = "8VLG910NQ20XV"

# Map of raw input -> canonical norm_partname
# The input token is normalized to uppercase and spaces are removed before lookup.
_EQUIV_ALIAS_MAP = {
    "8VLG910NQ20XV-L0OMZ": _CANON_EQUIV_KEY,
    "8VL8744NQ20XV-L0VGZ": _CANON_EQUIV_KEY,
    "8TLG912NQ20V-L0DZ":   _CANON_EQUIV_KEY,
}


# ------------------------------- Public API ----------------------------------

def normalize_partname(partname: Optional[str]) -> str:
    """
    Normalize a raw PartName into a 'norm_partname' used for grouping and model lookups.
    
    Normalization steps:
    1. Extract first token and convert to uppercase.
    2. Remove all whitespace.
    3. Check against explicit manual alias map.
    4. Normalize standard prefixes (e.g., 6CL -> 6L).
    5. Strip specific trailing package suffixes (-D, -L).
    
    Args:
        partname: The raw input PartName.
        
    Returns:
        The normalized partname string.
    """
    if partname is None:
        return ""

    pn_raw = str(partname).strip()
    if not pn_raw or pn_raw in ("NO_RECORD", "TIMEOUT", "UNKNOWN"):
        return ""

    token = _get_first_token(pn_raw)
    if not token:
        return ""

    # Sanitize: Upper and no internal spaces
    s = token.strip().upper()
    s = re.sub(r"\s+", "", s)

    # (A) Check targeted alias overrides
    if s in _EQUIV_ALIAS_MAP:
        return _EQUIV_ALIAS_MAP[s]

    # (B1) Prefix normalize <digits>(C|H)L -> <digits>L
    m = _PREFIX_NORM_RE.match(s)
    if m:
        s = f"{m.group('num')}L{m.group('rest')}"

    # (B2) Drop trailing -D / -L
    s = _TRAILING_D_L_RE.sub("", s)

    return s


def add_suffix_to_generic(generic: Optional[str], partname: Optional[str]) -> str:
    """
    Compose a unique identifier using the Generic name and its normalized PartName.
    
    Args:
        generic: The base product generic name.
        partname: The raw PartName to be normalized as a suffix.
        
    Returns:
        A string in the format "{Generic}_{norm_partname}".
    """
    gen = (generic or "").strip()
    norm = normalize_partname(partname)
    if not norm:
        return gen
    return f"{gen}_{norm}"


def main() -> None:
    """CLI utility for testing PartName normalization."""
    import argparse

    ap = argparse.ArgumentParser(description="Standalone utility to compute normalized {Generic}_{norm_partname}")
    ap.add_argument("-g", "--generic", required=True, help="Product generic string")
    ap.add_argument("-p", "--partname", required=True, help="Raw PartName string")
    args = ap.parse_args()

    print(add_suffix_to_generic(args.generic, args.partname))


if __name__ == "__main__":
    main()