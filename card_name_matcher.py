#!/usr/bin/env python3
"""
Card Name Matcher - Optimized Version

Fuzzy-match Yu-Gi-Oh! card names to canonical forms and check Legacy Pack status.
Optimized for speed with aggressive caching and minimal network requests.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
from urllib.parse import quote
import requests
from rapidfuzz import process, fuzz

# Configuration
YGOPRODECK_API = "https://db.ygoprodeck.com/api/v7/cardinfo.php"
CARDS_CACHE = Path("cache/ygo_cards_cache.json")
LEGACY_CACHE = Path("cache/legacy_cache.json")
REQUEST_TIMEOUT = 6.0
HEADERS = {"User-Agent": "ygo-collection-tool/1.0"}

# Module-level cache to avoid repeated file I/O
_CARDS_MAP_CACHE = None
_LEGACY_CACHE = None

def fetch_or_load_cards() -> Dict[str, Dict]:
    """Load canonical card names from cache or fetch from API"""
    global _CARDS_MAP_CACHE
    if _CARDS_MAP_CACHE is not None:
        return _CARDS_MAP_CACHE
    
    if CARDS_CACHE.exists():
        try:
            with CARDS_CACHE.open("r", encoding="utf-8") as f:
                _CARDS_MAP_CACHE = json.load(f)
            return _CARDS_MAP_CACHE
        except Exception:
            pass
    
    try:
        r = requests.get(YGOPRODECK_API, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        js = r.json()
        raw = js.get("data") or js.get("cards") or js
        mapping = {}
        for entry in raw:
            name = entry.get("name") or entry.get("cardname") or entry.get("card_name")
            if name:
                mapping[name] = {"id": entry.get("id"), "name": name}
        CARDS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with CARDS_CACHE.open("w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=1)
        _CARDS_MAP_CACHE = mapping
        return mapping
    except Exception:
        if CARDS_CACHE.exists():
            with CARDS_CACHE.open("r", encoding="utf-8") as f:
                _CARDS_MAP_CACHE = json.load(f)
            return _CARDS_MAP_CACHE
        raise

class CanonicalMatcher:
    """Fast fuzzy matcher using RapidFuzz"""
    def __init__(self, names):
        self.names = list(names)
    
    def match(self, query: str) -> Tuple[Optional[str], float]:
        """Return best matching canonical name and confidence score"""
        if not query or not self.names:
            return None, 0.0
        best = process.extractOne(query, self.names, scorer=fuzz.WRatio, score_cutoff=0)
        if best is None:
            return None, 0.0
        matched_name, score, _ = best
        return matched_name, float(score)

def load_legacy_cache() -> Dict[str, bool]:
    """Load legacy pack status cache from disk"""
    global _LEGACY_CACHE
    if _LEGACY_CACHE is not None:
        return _LEGACY_CACHE
    
    if LEGACY_CACHE.exists():
        try:
            data = json.loads(LEGACY_CACHE.read_text(encoding="utf-8"))
            _LEGACY_CACHE = {k: v.get("legacy", False) for k, v in data.items()}
            return _LEGACY_CACHE
        except Exception:
            pass
    _LEGACY_CACHE = {}
    return _LEGACY_CACHE

def save_legacy_cache(cache: Dict[str, bool]):
    """Persist legacy pack status cache to disk"""
    try:
        LEGACY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        data = {k: {"legacy": v} for k, v in cache.items()}
        LEGACY_CACHE.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass

def is_in_legacy_pack(card_name: str, cache: Dict[str, bool]) -> bool:
    """Check if card is in Legacy Pack via MasterDuelMeta scraping"""
    if card_name in cache:
        return cache[card_name]
    
    url = f"https://www.masterduelmeta.com/cards/{quote(card_name)}"
    try:
        r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        if r.status_code == 404:
            return False
        r.raise_for_status()
        in_legacy = "Legacy Pack" in r.text
        cache[card_name] = in_legacy
        save_legacy_cache(cache)
        return in_legacy
    except Exception:
        return False

def get_canonical_name_and_legacy_status(card_name: str) -> Tuple[str, bool]:
    """
    Match card name to canonical form and determine legacy pack status.
    Returns (canonical_name, is_legacy_pack)
    """
    cards_map = fetch_or_load_cards()
    if not cards_map:
        return "", False

    matcher = CanonicalMatcher(list(cards_map.keys()))
    best_name, score = matcher.match(card_name)
    if not best_name:
        return "", False

    legacy_cache = load_legacy_cache()
    in_legacy = is_in_legacy_pack(best_name, legacy_cache)
    return best_name, in_legacy

if __name__ == "__main__":
    """Command-line interface for card name matching"""
    if len(sys.argv) > 1:
        card_name = " ".join(sys.argv[1:])
        canonical, legacy = get_canonical_name_and_legacy_status(card_name)
        print(f"Canonical Name: {canonical}")
        print(f"Legacy Pack: {legacy}")
    else:
        print("Usage: python card_name_matcher.py <card name>")
