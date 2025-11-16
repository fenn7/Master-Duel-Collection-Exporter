#!/usr/bin/env python3
"""
ygo_match_legacy_verbose_urls.py

Fuzzy-match a Yu-Gi-Oh! card name and check MasterDuelMeta Legacy Pack + Master Duel rarity.
This variant prints every attempted MasterDuelMeta URL and the HTTP response details so you can monitor failures.

Usage:
    python ygo_match_legacy_verbose_urls.py "call of the hantd"
"""

import argparse
import json
import os
import sys
import time
import unicodedata
import re
from pathlib import Path
from typing import Dict, Tuple, Optional

import requests
from bs4 import BeautifulSoup
from rapidfuzz import process, fuzz

# ---------------------------------------------------------------------
# Configuration / filenames
# ---------------------------------------------------------------------
YGOPRODECK_API = "https://db.ygoprodeck.com/api/v7/cardinfo.php"
CARDS_CACHE = Path("cache/ygo_cards_cache.json")
LEGACY_CACHE = Path("cache/legacy_cache.json")

SCORER = fuzz.WRatio
REQUEST_TIMEOUT = 12.0
HEADERS = {"User-Agent": "ygo-collection-tool/1.0 (+https://github.com/you)"}

# Priority keywords for fallback rarity extraction (more specific -> earlier)
RARITY_KEYWORDS_PRIORITY = [
    "Secret Rare",
    "Ultimate Rare",
    "Ghost Rare",
    "Ultra Rare",
    "Super Rare",
    "Rare",
    "Common",
    "Gold Rare",
    "Prismatic Rare",
    "Parallel Rare",
    "Collector's Rare",
    "Diamond Rare",
    "Normal",
    "SR",
    "UR",
    "MR",
]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def normalize_text(s: str) -> str:
    s = s or ""
    s = s.strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = " ".join(s.split())
    return s.lower()


# ---------------------------------------------------------------------
# Fetch / cache canonical names from YGOPRODECK
# ---------------------------------------------------------------------
def fetch_or_load_cards() -> Dict[str, Dict]:
    if CARDS_CACHE.exists():
        try:
            with CARDS_CACHE.open("r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"[cards] Loaded {len(data)} entries from cache {CARDS_CACHE}")
            return data
        except Exception as e:
            print(f"[cards] Failed to read cache ({e}), will attempt fresh download.")

    print("[cards] Downloading canonical card list from YGOPRODeck...")
    try:
        r = requests.get(YGOPRODECK_API, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        js = r.json()
        raw = js.get("data") or js.get("cards") or js
        mapping = {}
        for entry in raw:
            name = entry.get("name") or entry.get("cardname") or entry.get("card_name")
            if not name:
                continue
            mapping[name] = {"id": entry.get("id"), "name": name, "raw": entry}
        CARDS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with CARDS_CACHE.open("w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=1)
        print(
            f"[cards] Downloaded and cached {len(mapping)} card entries to {CARDS_CACHE}"
        )
        return mapping
    except Exception as e:
        print(f"[cards] ERROR: failed download: {e}")
        if CARDS_CACHE.exists():
            print("[cards] Falling back to existing cache file.")
            with CARDS_CACHE.open("r", encoding="utf-8") as f:
                return json.load(f)
        raise


# ---------------------------------------------------------------------
# RapidFuzz matcher
# ---------------------------------------------------------------------
class CanonicalMatcher:
    def __init__(self, names):
        self.names = list(names)
        self._norm_cache = {n: normalize_text(n) for n in self.names}

    def match(self, query: str) -> Tuple[Optional[str], float]:
        if not query or not self.names:
            return None, 0.0
        best = process.extractOne(query, self.names, scorer=SCORER, score_cutoff=0)
        if best is None:
            return None, 0.0
        matched_name, score, _ = best
        return matched_name, float(score)


# ---------------------------------------------------------------------
# MasterDuelMeta: determine Legacy Pack membership AND Master Duel rarity
# (verbose: prints attempted URL and response info)
# ---------------------------------------------------------------------
def load_legacy_cache() -> Dict[str, Dict]:
    if LEGACY_CACHE.exists():
        try:
            return json.loads(LEGACY_CACHE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_legacy_cache(cache: Dict):
    try:
        LEGACY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        LEGACY_CACHE.write_text(
            json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8"
        )
    except Exception as e:
        print(f"[legacy] Warning: failed to save legacy cache: {e}")


def _extract_rarity_from_soup(soup: BeautifulSoup) -> Optional[str]:
    """Try multiple heuristics to extract rarity text from the MasterDuelMeta page DOM."""
    # 1) Find element(s) containing the word "Rarity" and parse nearby text
    rarity_nodes = soup.find_all(
        string=lambda t: t and re.search(r"\brarity\b", t, re.I)
    )
    for node in rarity_nodes:
        try:
            text = node.parent.get_text(separator=" ", strip=True)
            parts = re.split(r"Rarity[:\s-]*", text, flags=re.I)
            if len(parts) >= 2:
                candidate = parts[1].strip()
                candidate = re.split(r"[\n\r/–\-\|,]", candidate)[0].strip()
                if candidate:
                    return candidate
        except Exception:
            continue

    # 2) Look for table style: find table rows where the header or first cell includes 'Rarity'
    for th in soup.find_all(["th", "td"]):
        try:
            label = th.get_text(separator=" ", strip=True)
            if re.search(r"\brarity\b", label, re.I):
                sib = th.find_next_sibling()
                if sib:
                    txt = sib.get_text(separator=" ", strip=True)
                    if txt:
                        return txt.split("\n")[0].strip()
        except Exception:
            continue

    # 3) Fallback: scan full page text for priority keywords (most specific first)
    page_text = soup.get_text(separator=" ", strip=True)
    for kw in RARITY_KEYWORDS_PRIORITY:
        if re.search(re.escape(kw), page_text, re.I):
            return kw

    # 4) Nothing found
    return None


def is_in_legacy_pack_masterduelmeta(
    card_name: str, cache: Dict
) -> Tuple[bool, Optional[str], str]:
    """
    Checks MasterDuelMeta for legacy pack membership and returns (legacy_bool, rarity_or_None, detail_text).
    This verbose variant prints the attempted URL and the HTTP response status/resolved URL.
    """
    if card_name in cache:
        entry = cache[card_name]
        return (
            bool(entry.get("legacy", False)),
            entry.get("rarity"),
            entry.get("detail", ""),
        )

    from urllib.parse import quote

    url_encoded = quote(card_name)
    url = f"https://www.masterduelmeta.com/cards/{url_encoded}"

    # Print attempted URL for monitoring
    print(f"Attempting MasterDuelMeta URL: {url}")

    try:
        r = requests.get(
            url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True
        )
    except Exception as e:
        detail = f"network error when requesting {url}: {e}"
        print(f"Network error for URL {url}: {e}")
        # Do not cache on failure
        return False, None, detail

    # Print response status and final URL after redirects
    try:
        status = r.status_code
        final_url = getattr(r, "url", url)
        print(f"HTTP {status} - Resolved URL: {final_url}")
        # Print a short snippet of the body for debugging if non-200 (first 300 chars)
        if status != 200:
            snippet = (r.text or "")[:300].replace("\n", " ").replace("\r", " ")
            print(f"Response body snippet (first 300 chars): {snippet!s}")
    except Exception as e:
        print(f"Failed to inspect response: {e}")

    if r.status_code == 404:
        detail = f"card page not found (404) for attempted URL: {url}"
        print(f"404 Not Found for URL: {url}")
        # Do not cache on failure
        return False, None, detail

    try:
        r.raise_for_status()
    except Exception as e:
        detail = f"HTTP error {r.status_code} for {url}: {e}"
        print(f"HTTP error {r.status_code} for URL: {url} - {e}")
        # Do not cache on failure
        return False, None, detail

    html = r.text

    # quick detect legacy
    in_legacy = "Legacy Pack" in html or "Legacy pack" in html or "Legacy Pack" in html

    # parse DOM to try to extract rarity
    soup = BeautifulSoup(html, "lxml")
    rarity = _extract_rarity_from_soup(soup)

    detail = f"resolved_url={r.url}; http_status={r.status_code}"
    if in_legacy:
        detail += "; Legacy Pack text found"
    if rarity:
        detail += f"; rarity={rarity}"
    if not detail:
        detail = "no legacy/rarity info found (heuristics)."

    cache[card_name] = {"legacy": bool(in_legacy), "rarity": rarity, "detail": detail}
    save_legacy_cache(cache)
    return bool(in_legacy), rarity, detail


# ---------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Fuzzy-match a Yu-Gi-Oh! card name and check MasterDuelMeta Legacy Pack + MD rarity (verbose URLs)."
    )
    p.add_argument(
        "query", nargs="+", help="Input string to match to a canonical card name"
    )
    p.add_argument(
        "--no-network",
        action="store_true",
        help="Do not attempt network fetch (use caches only)",
    )
    args = p.parse_args()

    query_raw = " ".join(args.query).strip()
    if not query_raw:
        print("Empty query.")
        sys.exit(2)

    print(f"[input] Query: {query_raw!r}")

    print(f"[input] Query: {query_raw!r}")

    if args.no_network and not CARDS_CACHE.exists():
        print("[error] --no-network requested but no local cards cache available.")
        sys.exit(2)

    cards_map = (
        fetch_or_load_cards()
        if not args.no_network
        else json.loads(CARDS_CACHE.read_text(encoding="utf-8"))
    )
    canonical_names = list(cards_map.keys())
    if not canonical_names:
        print("[error] No canonical card names available.")
        sys.exit(1)

    matcher = CanonicalMatcher(canonical_names)
    best_name, score = matcher.match(query_raw)
    if not best_name:
        print("[result] No match found.")
        sys.exit(0)

    print(f"[match] Best canonical name: {best_name!r}  score={score:.2f} (0-100)")

    legacy_cache = load_legacy_cache()
    in_legacy, rarity, detail = is_in_legacy_pack_masterduelmeta(
        best_name, legacy_cache
    )
    print(f"[legacy] In Legacy Pack (MasterDuelMeta): {in_legacy}  detail: {detail}")
    print(f"[rarity] Master Duel rarity (heuristic): {rarity!r}")


if __name__ == "__main__":
    main()
