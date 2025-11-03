#!/usr/bin/env python3
"""
masterduel_collection_scraper.py

Base runnable program to:
 - detect the card-collection grid area in the active Master Duel window
 - repeatedly screenshot that region and crop thumbnails
 - OCR card names + counts from each thumbnail
 - fuzzy-match names to a canonical YGOJSON database
 - automate scrolling until the entire collection has been captured
 - export results to CSV and optionally Google Sheets

This is a base and needs tuning per user resolution, language, and UI skin.
Test locally, do not upload screenshots without consent.
"""

import time
import json
import os
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
from PIL import Image
import pytesseract
from rapidfuzz import process, fuzz
import mss
import pyautogui
import pygetwindow as gw
import pandas as pd
import requests

# Optional imports - only used if available
try:
    from skimage.metrics import structural_similarity as ssim
except Exception:
    ssim = None

# ---------- Configuration ----------
# Path to local YGOJSON (set to your file) or let AUTO_FETCH_YGOJSON = True
YGOJSON_PATH = "ygo_db.json"
AUTO_FETCH_YGOJSON = True
YGOJSON_URL = "https://raw.githubusercontent.com/SalvationDevelopment/YGO-DB/master/ygo_db.json"

# Tesseract config (adjust if tesseract not in PATH)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Window title substring for Master Duel - adjust to your environment
WINDOW_TITLE_KEYWORD = "masterduel"

# Heuristics and thresholds (tweak for your setup)
MIN_CARD_AREA = 2500            # minimal bounding box area for detected thumbnail
MAX_CARD_AREA_RATIO = 0.12      # maximal area ratio relative to full region to be considered a thumbnail
CARD_ASPECT_MIN = 0.6
CARD_ASPECT_MAX = 1.6
OCR_CONF_THRESHOLD = 60         # fuzzy match score threshold for automatic mapping
SCROLL_PIXELS = -700            # negative for scrolling down with pyautogui.scroll
STABILIZATION_WAIT = 0.45       # seconds to wait after scroll before capture
MAX_CONSECUTIVE_NO_NEW = 4      # stop if no new cards found after this many scrolls
OUTPUT_CSV = "collection_output.csv"

# For comparing screenshots to detect end of list or looping
IMAGE_DIFF_THRESHOLD = 0.03     # fraction of changed pixels considered as "changed"
USE_SSIM = True                 # use SSIM if available for more robust equality checks
# -----------------------------------

# Data containers
@dataclass
class CardRecord:
    canonical_id: Optional[str]
    canonical_name: str
    raw_name: str
    count: int
    confidence: float
    bbox: Tuple[int,int,int,int]   # relative to grid crop (x,y,w,h)

# ---------- Utilities ----------

def load_ygojson_web_only(timeout: int = 30, max_retries: int = 3, retry_backoff: float = 1.0) -> List[Dict[str, Any]]:
    """
    Always fetch a canonical card DB from the web and return a flat list of entries:
      [{ "name": <card name>, "id": <card id> }, ...]
    Tries multiple known endpoints (YGOPRODeck primary, iconmaster as fallback).
    Raises RuntimeError if all endpoints fail to return parseable data.
    """
    # Primary and fallback endpoints (order matters)
    urls = [
        "https://db.ygoprodeck.com/api/v7/cardinfo.php",  # primary (returns {"data":[...]}).
        "https://raw.githubusercontent.com/iconmaster5326/YGOJSON/v1/aggregate/cards.json",  # fallback aggregate (may be large / sometimes out-of-date).
    ]

    session = requests.Session()
    session.headers.update({
        "User-Agent": "MDM-Collection-Exporter/1.0 (+https://example.example)",  # set something sensible
        "Accept": "application/json, text/json, */*;q=0.1"
    })

    def try_parse_loaded(data: Any) -> List[Dict[str, Any]]:
        flat = []
        # Case 1: direct list of card dicts
        if isinstance(data, list):
            for c in data:
                if not isinstance(c, dict):
                    continue
                name = c.get("name") or c.get("card_name") or c.get("english_name")
                cid = c.get("id") or c.get("card_id") or name
                if name:
                    flat.append({"name": name, "id": cid})
            if flat:
                return flat

        # Case 2: {"data": [...]} typical of YGOPRODeck
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            for c in data["data"]:
                if not isinstance(c, dict):
                    continue
                name = c.get("name") or c.get("card_name")
                cid = c.get("id") or c.get("card_id") or name
                if name:
                    flat.append({"name": name, "id": cid})
            if flat:
                return flat

        # Case 3: {"cards": [...]} or other common shapes
        if isinstance(data, dict) and "cards" in data and isinstance(data["cards"], list):
            for c in data["cards"]:
                if not isinstance(c, dict):
                    continue
                name = c.get("name") or c.get("card_name")
                cid = c.get("id") or name
                if name:
                    flat.append({"name": name, "id": cid})
            if flat:
                return flat

        # Case 4: mapping of keys -> card dicts { id: { "name": ... }, ... }
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict) and "name" in v:
                    flat.append({"name": v.get("name"), "id": v.get("id", k)})
            if flat:
                return flat

        # If no parse succeeded, raise to let caller try next URL
        raise ValueError("Unable to parse canonical DB structure from fetched JSON.")

    last_exc = None
    for url in urls:
        for attempt in range(1, max_retries + 1):
            try:
                # GET with timeout
                resp = session.get(url, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                parsed = try_parse_loaded(data)
                # success
                print(f"[load_ygojson_web_only] Fetched {len(parsed)} entries from {url}")
                return parsed
            except Exception as e:
                last_exc = e
                wait = retry_backoff * (2 ** (attempt - 1))
                print(f"[load_ygojson_web_only] Attempt {attempt} failed for {url}: {e!r}; retrying in {wait:.1f}s..." if attempt < max_retries else f"[load_ygojson_web_only] Attempt {attempt} failed for {url} and no retries left: {e!r}")
                if attempt < max_retries:
                    time.sleep(wait)
                else:
                    # try next URL after exhausting retries for this URL
                    break

    # If all URLs exhausted
    raise RuntimeError(f"Unable to fetch/parse a canonical card DB from web sources. Last error: {last_exc!r}")

def load_ygojson(path: str, auto_fetch: bool = True) -> List[Dict[str, Any]]:
    """Load YGOJSON file with canonical card names and ids."""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif auto_fetch:
        print("YGOJSON not found locally. Attempting to fetch from remote...")
        r = requests.get(YGOJSON_URL, timeout=20)
        r.raise_for_status()
        data = r.json()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    else:
        raise FileNotFoundError("YGOJSON not found and auto_fetch disabled.")
    # Data shape varies by source. Expected: list of cards with 'name' and 'id' or 'id' may be 'name' as key.
    # Normalize into a list of (name, id)
    flat = []
    if isinstance(data, list):
        for c in data:
            name = c.get("name") or c.get("card_name") or c.get("english_name")
            cid = c.get("id") or c.get("card_id") or name
            if name:
                flat.append({"name": name, "id": cid})
    elif isinstance(data, dict):
        # try common variations
        cards = data.get("data") or data.get("cards") or []
        if isinstance(cards, list):
            for c in cards:
                name = c.get("name") or c.get("card_name")
                cid = c.get("id") or name
                if name:
                    flat.append({"name": name, "id": cid})
        else:
            # fallback: iterate keys
            for k,v in data.items():
                if isinstance(v, dict) and ("name" in v):
                    flat.append({"name": v["name"], "id": v.get("id", k)})
    else:
        raise ValueError("Unexpected ygojson structure")
    print(f"Loaded {len(flat)} canonical entries from YGOJSON")
    return flat

import re
def normalize_title(s: str) -> str:
    return re.sub(r'[^0-9a-z]', '', (s or "").lower())

def find_game_window(keyword: str = WINDOW_TITLE_KEYWORD) -> Optional[gw.Win32Window]:
    """Find the Master Duel window using pygetwindow. Returns first match or None."""
    try:
        wins = gw.getWindowsWithTitle(normalize_title(keyword))
    except Exception:
        # fallback: iterate all windows looking for substring
        all_windows = gw.getAllWindows()
        wins = [w for w in all_windows if keyword.lower() in w.title.lower()]
    if not wins:
        print("Could not find game window by title. Please ensure it is open and not minimized.")
        return None
    win = wins[0]
    if win.isMinimized:
        try:
            win.restore()
            time.sleep(0.3)
        except Exception:
            pass
    # bring to front
    try:
        win.activate()
        time.sleep(0.2)
    except Exception:
        pass
    return win

def grab_region(region: Tuple[int,int,int,int]) -> np.ndarray:
    """
    Robust region grab using mss -> numpy -> OpenCV conversion.
    Region is (left, top, width, height). Returns BGR image (np.ndarray).
    """
    left, top, width, height = region
    import mss
    try:
        with mss.mss() as sct:
            monitor = {"left": int(left), "top": int(top), "width": int(width), "height": int(height)}
            sct_img = sct.grab(monitor)
            # Convert to numpy array. mss returns BGRA on most Windows setups.
            arr = np.array(sct_img)  # shape: (height, width, 3) or (height, width, 4)
            # If arr has 4 channels, convert BGRA -> BGR; if 3, assume it's already BGR-like
            if arr.ndim == 3 and arr.shape[2] == 4:
                img_bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            elif arr.ndim == 3 and arr.shape[2] == 3:
                # Some platforms may provide RGB ordering; try converting from RGB->BGR
                # We assume arr is RGB-like; convert to BGR for OpenCV
                img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:
                # unexpected shape - try forcing into BGR
                img_bgr = arr.copy()
            return img_bgr
    except Exception as e:
        # helpful message and re-raise so existing flow can handle failures
        print(f"[grab_region] Screen capture failed for region {region}: {e}")
        raise


# ---------- Grid detection methods ----------

def detect_grid_template(image: np.ndarray, templates: List[np.ndarray]) -> Optional[Tuple[int,int,int,int]]:
    """If you can supply UI templates (header, search bar), try template matching.
    Returns top-left + w,h of the computed grid region relative to image, or None."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h_img, w_img = gray.shape
    for tpl in templates:
        try:
            tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY) if tpl.ndim==3 else tpl
            res = cv2.matchTemplate(gray, tpl_gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > 0.65:
                tx, ty = max_loc
                tw, th = tpl_gray.shape[1], tpl_gray.shape[0]
                # heuristic: assume grid starts a fixed offset below template
                grid_left = max(0, tx - 8)
                grid_top = ty + th + 6
                grid_w = min(w_img - grid_left, w_img - grid_left - 6)
                grid_h = min(h_img - grid_top, h_img - grid_top - 6)
                return (grid_left, grid_top, grid_w, grid_h)
        except Exception:
            continue
    return None

def detect_grid_contours(image: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    """Heuristic: detect many small rectangular thumbnails and compute their enclosing bounding box."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    dil = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = gray.shape
    candidates = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = w*h
        if area < MIN_CARD_AREA:
            continue
        if area > MAX_CARD_AREA_RATIO * (w_img*h_img):
            continue
        aspect = w/h if h>0 else 0
        if not (CARD_ASPECT_MIN < aspect < CARD_ASPECT_MAX):
            continue
        candidates.append((x,y,w,h,area))

    if not candidates:
        # fallback morphological approach to find the large area containing many small items
        morph = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (40,40)))
        diff = cv2.absdiff(morph, blurred)
        _, th = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)
        cnts2, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0
        for cnt in cnts2:
            x,y,w,h = cv2.boundingRect(cnt)
            a = w*h
            if a > best_area:
                best_area = a
                best = (x,y,w,h)
        if best and best_area > 0.02 * (w_img*h_img):
            return best
        return None

    # Compute bounding box that encloses most candidates
    xs = [c[0] for c in candidates]
    ys = [c[1] for c in candidates]
    ws = [c[2] for c in candidates]
    hs = [c[3] for c in candidates]
    bx = max(min(xs)-6, 0)
    by = max(min(ys)-6, 0)
    bx2 = min(max([x+w for x,w in zip(xs,ws)])+6, w_img)
    by2 = min(max([y+h for y,h in zip(ys,hs)])+6, h_img)
    return (bx, by, bx2-bx, by2-by)

# ---------- Thumbnail extraction + OCR ----------

def split_grid_to_thumbs(grid_img: np.ndarray, expected_rows=None, expected_cols=None) -> List[Tuple[np.ndarray, Tuple[int,int,int,int]]]:
    """
    Try to tile the grid into thumbnails using detected contours or regular grid split.
    Returns list of (thumb_image, bbox) where bbox is relative to grid_img.
    """
    # Try contour-based detection first
    gray = cv2.cvtColor(grid_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3,3), 0)
    edges = cv2.Canny(blurred, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    dil = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = gray.shape
    boxes = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = w*h
        if area < MIN_CARD_AREA:
            continue
        aspect = w/h if h>0 else 0
        if not (CARD_ASPECT_MIN < aspect < CARD_ASPECT_MAX):
            continue
        boxes.append((x,y,w,h))
    if len(boxes) >= 6:
        # remove nested/overlapping boxes heuristically
        boxes_sorted = sorted(boxes, key=lambda b: b[2]*b[3], reverse=True)
        kept = []
        for b in boxes_sorted:
            x,y,w,h = b
            overlap = False
            for k in kept:
                kx,ky,kw,kh = k
                ix = max(0, min(x+w, kx+kw) - max(x, kx))
                iy = max(0, min(y+h, ky+kh) - max(y, ky))
                inter = ix*iy
                if inter / float(w*h + kw*kh - inter + 1e-8) > 0.3:
                    overlap = True
                    break
            if not overlap:
                kept.append(b)
        boxes = sorted(kept, key=lambda b:(b[1], b[0]))  # top-to-bottom, left-to-right
        thumbs = []
        for (x,y,w,h) in boxes:
            thumb = grid_img[y:y+h, x:x+w].copy()
            thumbs.append((thumb, (x,y,w,h)))
        return thumbs

    # If contour method failed, try tiling into a grid
    # Heuristic: try common columns 5-7 and rows accordingly
    for cols in range(6, 3, -1):  # try 6,5,4
        tile_w = w_img // cols
        rows = int(np.ceil(h_img / (tile_w + 1)))  # approximate
        thumbs = []
        for r in range(rows):
            for c in range(cols):
                x = c * tile_w
                y = r * tile_w
                w = tile_w
                h = tile_w
                if x + w > w_img:
                    w = w_img - x
                if y + h > h_img:
                    h = h_img - y
                if w <= 5 or h <= 5:
                    continue
                thumb = grid_img[y:y+h, x:x+w].copy()
                thumbs.append((thumb, (x,y,w,h)))
        if len(thumbs) >= 6:
            return thumbs

    return []

def ocr_name_and_count(thumb_img: np.ndarray, lang: str = "eng") -> Tuple[str, int, float]:
    """
    Run OCR on a thumbnail to extract the card name and the count badge.
    Returns raw_name, count, confidence_score
    """
    h, w = thumb_img.shape[:2]
    # preprocess to emphasize text
    # crop top portion where name usually resides (tweak per language/skin)
    name_h = int(h * 0.22)
    name_region = thumb_img[5:name_h, 6:w-6]  # small margins
    name_gray = cv2.cvtColor(name_region, cv2.COLOR_BGR2GRAY)
    name_gray = cv2.resize(name_gray, (name_gray.shape[1]*2, name_gray.shape[0]*2), interpolation=cv2.INTER_CUBIC)
    name_gray = cv2.adaptiveThreshold(name_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # OCR config: single line of text
    custom_config = r'--oem 3 --psm 7'
    raw_name = pytesseract.image_to_string(name_gray, lang=lang, config=custom_config)
    raw_name = raw_name.strip().replace("\n", " ").replace("\x0c", "").strip()

    # Count badge at bottom-right corner
    count = 1
    cr = thumb_img[max(0, h-34):h-6, max(0, w-34):w-6].copy()
    cr_gray = cv2.cvtColor(cr, cv2.COLOR_BGR2GRAY)
    cr_gray = cv2.resize(cr_gray, (cr_gray.shape[1]*2, cr_gray.shape[0]*2), interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(cr_gray, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cfg_count = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
    try:
        txt = pytesseract.image_to_string(th, config=cfg_count)
        txt = txt.strip()
        if txt.isdigit():
            count = int(txt)
    except Exception:
        count = 1

    # confidence placeholder: we can compute fuzzy match confidence later
    return raw_name, count, 0.0

# ---------- Matching to canonical DB ----------

class CanonicalMatcher:
    def __init__(self, entries: List[Dict[str, Any]]):
        # build lookup structures
        self.names = [e["name"] for e in entries]
        self.map = {e["name"]: e.get("id", e["name"]) for e in entries}

    def match(self, raw_name: str) -> Tuple[Optional[str], str, float]:
        """Return (canonical_id, canonical_name, score)"""
        if not raw_name:
            return None, raw_name, 0.0
        res = process.extractOne(raw_name, self.names, scorer=fuzz.ratio)
        if res:
            name, score, idx = res
            cid = self.map.get(name, name)
            return cid, name, float(score)
        return None, raw_name, 0.0

# ---------- Main capture and loop ----------

def detect_grid_for_window(win) -> Optional[Tuple[int,int,int,int]]:
    """
    Safely try to detect grid region. If window has negative coordinates, try moving it
    to positive coordinates first (best-effort).
    """
    left, top, width, height = win.left, win.top, win.width, win.height

    # If window coordinates are negative (on another monitor), try to move it to primary
    if left < 0 or top < 0:
        print(f"[detect] Window at negative coords ({left},{top}). Attempting to move window to primary monitor (8,8).")
        try:
            # small move to primary monitor - best-effort, may raise on some platforms
            win.moveTo(8, 8)
            time.sleep(0.35)
            left, top, width, height = win.left, win.top, win.width, win.height
            print(f"[detect] Window moved to {left},{top} (size {width}x{height})")
        except Exception as e:
            print(f"[detect] Could not move window: {e}. Proceeding with original coords.")

    # Final attempt to grab the full window region
    try:
        img = grab_region((left, top, width, height))
    except Exception as e:
        print(f"[detect] Full-window grab failed: {e}")
        return None

    # template list load as before...
    templates = []
    tpl_dir = Path("templates")
    if tpl_dir.exists():
        for p in tpl_dir.glob("*.png"):
            tpl = cv2.imread(str(p))
            if tpl is not None:
                templates.append(tpl)

    # Try template matching, else contours (same functions as original)
    if templates:
        res = detect_grid_template(img, templates)
        if res:
            gl, gt, gw, gh = res
            return (left + gl, top + gt, gw, gh)

    res = detect_grid_contours(img)
    if res:
        gl, gt, gw, gh = res
        return (left + gl, top + gt, gw, gh)

    return None

# ---------- Optional Google Sheets upload (fill in credentials) ----------
def export_to_gsheet(records: List[CardRecord], sheet_name: str, creds_json: str):
    """
    Optional: write to Google Sheets. creds_json is path to service account JSON.
    Requires gspread and oauth2client.
    """
    try:
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials
    except Exception:
        print("gspread not installed. Install 'gspread oauth2client' to enable Google Sheets export.")
        return
    scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name(creds_json, scope)
    client = gspread.authorize(creds)
    sh = client.create(sheet_name)
    ws = sh.sheet1
    headers = ["canonical_id","canonical_name","raw_name","count","confidence","bbox_x","bbox_y","bbox_w","bbox_h"]
    ws.append_row(headers)
    for r in records:
        ws.append_row([r.canonical_id, r.canonical_name, r.raw_name, r.count, r.confidence, r.bbox[0], r.bbox[1], r.bbox[2], r.bbox[3]])
    print("Uploaded sheet:", sh.url)

# ---------- Entrypoint ----------

def main():
    print("Master Duel collection scraper - starting")
    # Load canonical DB
    entries = load_ygojson_web_only()
    matcher = CanonicalMatcher(entries)

    # Find game window
    win = find_game_window(WINDOW_TITLE_KEYWORD)
    if not win:
        print("Could not find Master Duel window. Exiting.")
        return

    print(f"Detected window: {win.title} @ {win.left},{win.top} size {win.width}x{win.height}")
    print("Detecting collection grid region automatically...")
    grid = detect_grid_for_window(win)
    if not grid:
        print("Automatic detection failed. Please create a template image of the collection header and place it into ./templates/")
        # as fallback ask user to manually provide region
        try:
            print("Please move the mouse to the top-left of the grid and press Enter...")
            input()
            lx, ly = pyautogui.position()
            print("Now move the mouse to the bottom-right of the grid and press Enter...")
            input()
            rx, ry = pyautogui.position()
            grid = (min(lx, rx), min(ly, ry), abs(rx-lx), abs(ry-ly))
        except Exception:
            print("Manual region selection failed. Exiting.")
            return

    print(f"Using grid region: {grid}")
    print("Starting capture loop. Do not use the mouse or keyboard to interact with the game while capture runs.")
    records = run_capture_loop(grid, matcher)
    export_to_csv(records)

import cv2
import numpy as np

def images_similar(a: np.ndarray, b: np.ndarray, ssim_threshold: float = 0.995, diff_threshold: float = 0.03) -> bool:
    """
    Return True when images `a` and `b` are visually very similar.

    - a, b: BGR numpy arrays (as returned by grab_region / cv2).
    - If shapes differ, `b` is resized to `a` using cv2.INTER_LINEAR.
    - First tries SSIM (if scikit-image is installed). If unavailable, falls back to
      normalized pixel absolute-difference fraction.
    - ssim_threshold: closeness threshold for SSIM (1.0 is identical).
    - diff_threshold: fraction of differing pixels allowed for the fallback method.

    Usage: replace previous images_similar calls with this function.
    """
    if a is None or b is None:
        return False

    # Ensure both are numpy arrays
    a = np.array(a)
    b = np.array(b)

    # If dimensions differ, resize b to match a
    if a.shape != b.shape:
        try:
            b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)
        except Exception:
            return False

    # Convert to grayscale for comparison
    try:
        grayA = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        grayB = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    except Exception:
        # if conversion fails (unexpected channels), compare on raw arrays
        grayA = a if a.ndim == 2 else cv2.cvtColor(a, cv2.COLOR_RGBA2GRAY) if a.shape[2] == 4 else a[...,0]
        grayB = b if b.ndim == 2 else cv2.cvtColor(b, cv2.COLOR_RGBA2GRAY) if b.shape[2] == 4 else b[...,0]

    # Try SSIM if available
    try:
        from skimage.metrics import structural_similarity as ssim
        score, _ = ssim(grayA, grayB, full=True)
        return score >= ssim_threshold
    except Exception:
        # Fall back to pixel-diff fraction
        diff = cv2.absdiff(grayA, grayB)
        # Count pixels that differ above a small per-pixel threshold to ignore compression/noise
        _, th = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)
        nonzero = np.count_nonzero(th)
        total = th.size
        frac = nonzero / float(total)
        return frac <= diff_threshold

import time
from collections import Counter
from typing import List

def live_progress_summary(records: List, iteration: int, start_time: float,
                          recent_new: List = None, top_n: int = 6,
                          ocr_conf_threshold: float = OCR_CONF_THRESHOLD):
    """
    Print a concise live progress summary.

    - records: list of CardRecord (dataclass) or dicts with 'canonical_name', 'raw_name', 'count', 'confidence'
    - iteration: current loop iteration counter (int)
    - start_time: time.time() value when capture started
    - recent_new: optional list of recently added record objects (to show last found names)
    - top_n: how many top cards to display
    - ocr_conf_threshold: threshold below which a detection is considered low-confidence
    """
    now = time.time()
    elapsed = now - start_time
    hh = int(elapsed // 3600)
    mm = int((elapsed % 3600) // 60)
    ss = int(elapsed % 60)
    elapsed_s = f"{hh:d}h {mm:02d}m {ss:02d}s" if hh else f"{mm:02d}m {ss:02d}s"

    total_unique = len(records)
    total_copies = 0
    low_conf_count = 0
    name_list = []

    for r in records:
        # support both dataclass CardRecord and plain dicts
        cnt = 1
        conf = 0.0
        name = None
        if hasattr(r, "count"):
            try:
                cnt = int(r.count or 1)
            except Exception:
                cnt = 1
        elif isinstance(r, dict):
            cnt = int(r.get("count", 1))
        # confidence
        if hasattr(r, "confidence"):
            conf = float(getattr(r, "confidence") or 0.0)
        elif isinstance(r, dict):
            conf = float(r.get("confidence", 0.0) or 0.0)
        # name
        if hasattr(r, "canonical_name") and getattr(r, "canonical_name"):
            name = r.canonical_name
        elif hasattr(r, "raw_name") and getattr(r, "raw_name"):
            name = r.raw_name
        elif isinstance(r, dict):
            name = r.get("canonical_name") or r.get("raw_name") or r.get("name")

        total_copies += cnt
        if conf < ocr_conf_threshold:
            low_conf_count += 1
        if name:
            name_list.append((name, cnt))

    # compute top N names by copies
    counter = Counter()
    for n, c in name_list:
        counter[n] += int(c)
    top_items = counter.most_common(top_n)

    # recent found names
    recent_names = []
    if recent_new:
        for r in recent_new:
            # extract friendly name
            if hasattr(r, "canonical_name") and getattr(r, "canonical_name"):
                recent_names.append(getattr(r, "canonical_name"))
            elif hasattr(r, "raw_name") and getattr(r, "raw_name"):
                recent_names.append(getattr(r, "raw_name"))
            elif isinstance(r, dict):
                recent_names.append(r.get("canonical_name") or r.get("raw_name") or r.get("name"))

    rate_per_min = (total_unique / (elapsed / 60)) if elapsed > 5 else float(total_unique)

    # Print a tidy block
    print("=" * 60)
    print(f"[Progress] iteration={iteration}  elapsed={elapsed_s}  unique={total_unique}  copies={total_copies}")
    print(f"[Progress] low-confidence detections (score < {ocr_conf_threshold}): {low_conf_count}")
    print(f"[Progress] avg discovery rate: {rate_per_min:.2f} unique cards / min")
    if top_items:
        tops = ", ".join([f"{i+1}. {name} x{count}" for i, (name, count) in enumerate(top_items)])
        print(f"[Top {len(top_items)}] {tops}")
    if recent_names:
        print(f"[Recent found] " + ", ".join(recent_names[:8]))
    else:
        print("[Recent found] (none this iteration)")
    print("=" * 60)


def run_capture_loop(grid_region: Tuple[int,int,int,int],
                     matcher,
                     max_no_new: int = MAX_CONSECUTIVE_NO_NEW,
                     stabilization_wait: float = STABILIZATION_WAIT,
                     checkpoint_every: int = 50,
                     checkpoint_path: str = "collection_checkpoint.csv"):
    """
    Main capture+scroll loop.
    - grid_region: (left, top, width, height) in screen coords.
    - matcher: CanonicalMatcher instance with .match(raw_name) -> (id,name,score)
    - Returns list of CardRecord objects (same dataclass used elsewhere).
    """
    import time
    import hashlib
    import pandas as pd
    import pyautogui

    seen_keys = set()
    records = []
    consecutive_no_new = 0
    prev_top_snapshot = None
    iterations = 0

    print(f"[run_capture_loop] Starting capture loop on region={grid_region}")

    start_time = time.time()
    try:
        while True:
            iterations += 1
            left, top, width, height = grid_region
            # Capture current grid region
            try:
                crop = grab_region(grid_region)
            except Exception as e:
                print(f"[run_capture_loop] grab_region failed: {e}. Retrying after short sleep.")
                time.sleep(0.5)
                continue

            # Optionally show debug info or save debug image occasionally
            # cv2.imwrite(f"debug_full_{iterations:04d}.png", crop)

            # Split into thumbnail candidates
            thumbs = split_grid_to_thumbs(crop)
            new_found = 0

            for thumb_img, bbox in thumbs:
                raw_name, count, _ = ocr_name_and_count(thumb_img)
                cid, cname, score = matcher.match(raw_name)
                # Use canonical id if available, else canonical name; include count to avoid merging different counts
                key = (cid or cname or raw_name, int(count))
                if key not in seen_keys:
                    seen_keys.add(key)
                    rec = CardRecord(canonical_id=cid, canonical_name=cname, raw_name=raw_name, count=int(count), confidence=score, bbox=bbox)
                    records.append(rec)
                    new_found += 1

            # Top-slice similarity to detect repeated content after scroll
            top_row_h = min(120, crop.shape[0] // 6)
            top_slice = crop[0:top_row_h, :, :].copy()

            if prev_top_snapshot is not None and images_similar(prev_top_snapshot, top_slice):
                # top repeated; if also no new items, treat as no-new
                if new_found == 0:
                    consecutive_no_new += 1
            else:
                # content changed
                if new_found == 0:
                    # still treat as possibly no-new but reset less aggressively
                    consecutive_no_new += 0
                else:
                    consecutive_no_new = 0

            prev_top_snapshot = top_slice

            print(f"[capture #{iterations}] thumbs={len(thumbs)} new={new_found} total={len(records)} consecutive_no_new={consecutive_no_new}")
            recent_new = records[-new_found:] if new_found > 0 else []
            
            live_progress_summary(records, iterations, start_time, recent_new)

            # Periodic checkpoint save to CSV in case of crash
            if iterations % checkpoint_every == 0 and records:
                try:
                    rows = []
                    for r in records:
                        rows.append({
                            "canonical_id": r.canonical_id,
                            "canonical_name": r.canonical_name,
                            "raw_name": r.raw_name,
                            "count": r.count,
                            "confidence": r.confidence,
                            "bbox_x": r.bbox[0],
                            "bbox_y": r.bbox[1],
                            "bbox_w": r.bbox[2],
                            "bbox_h": r.bbox[3],
                        })
                    pd.DataFrame(rows).to_csv(checkpoint_path, index=False, encoding="utf-8-sig")
                    print(f"[run_capture_loop] Checkpoint saved to {checkpoint_path}")
                except Exception as e:
                    print(f"[run_capture_loop] Failed to write checkpoint: {e}")

            # Decide whether to stop
            if consecutive_no_new >= max_no_new:
                print("[run_capture_loop] Reached max consecutive no-new scrolls -> stopping capture.")
                break

            # Move mouse into grid so scroll applies
            gx, gy, gw, gh = grid_region
            cx = int(gx + gw // 2)
            cy = int(gy + gh - 80)
            try:
                pyautogui.moveTo(cx, cy, duration=0.12)
            except Exception:
                # ignore movement errors, still attempt to scroll
                pass

            # Try scrolling; if it fails, fallback to PageDown
            try:
                pyautogui.scroll(SCROLL_PIXELS)
            except Exception as e:
                print(f"[run_capture_loop] pyautogui.scroll failed: {e}; falling back to PageDown key.")
                try:
                    pyautogui.press("pagedown")
                except Exception as e2:
                    print(f"[run_capture_loop] PageDown fallback also failed: {e2}; aborting.")
                    break

            # Wait for UI to stabilize and let the new content render
            time.sleep(stabilization_wait)

    except KeyboardInterrupt:
        print("[run_capture_loop] KeyboardInterrupt received — stopping gracefully.")

    # Final de-duplication pass: collapse multiple entries of same canonical id summing counts
    aggregated = {}
    for r in records:
        key = (r.canonical_id or r.canonical_name or r.raw_name)
        if key not in aggregated:
            aggregated[key] = {"canonical_id": r.canonical_id, "canonical_name": r.canonical_name, "raw_name": r.raw_name, "count": r.count, "confidence": r.confidence}
        else:
            aggregated[key]["count"] = aggregated[key].get("count", 0) + r.count
            # keep max confidence
            aggregated[key]["confidence"] = max(aggregated[key]["confidence"], r.confidence)

    final_records = []
    for k, v in aggregated.items():
        final_records.append(CardRecord(canonical_id=v["canonical_id"], canonical_name=v["canonical_name"] or k, raw_name=v["raw_name"], count=int(v["count"]), confidence=float(v["confidence"]), bbox=(0,0,0,0)))

    print(f"[run_capture_loop] Finished. {len(final_records)} unique card entries aggregated from {len(records)} raw detections.")
    return final_records
    
    # Optional Google Sheets: uncomment and provide credentials file
    # export_to_gsheet(records, "MasterDuel Collection", "path_to_service_account.json")

if __name__ == "__main__":
    main()
