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

def ocr_description_zone_card_info(desc_zone_img: np.ndarray, lang: str = "eng", card_number: int = 0, row_number: int = 1) -> Tuple[str, int]:
    """
    Extract card name from the top area and count from bottom right of description zone.
    Returns (card_name, count)
    """
    if desc_zone_img is None or desc_zone_img.size == 0:
        return "", 1
    
    h, w = desc_zone_img.shape[:2]
    
    # Extract card name from top area, EXCLUDING the attribute symbol on the right
    name_h = int(h * 0.25)  # Top 25% for card name
    
    # FIXED: Exclude attribute symbol circle on right side
    # The attribute symbol appears to be in a circle at the right end of the name box
    # Exclude roughly 15% from the right side to avoid the symbol
    symbol_margin = int(w * 0.15)  # 15% margin from right for attribute symbol
    left_margin = int(w * 0.02)    # 2% margin from left
    top_margin = int(name_h * 0.1)  # 10% margin from top
    
    # REFINED: Further reduce capture area to focus only on text box
    # Reduce height by 30% and width by 1.7% of current height to exclude external symbols
    # CENTER the reductions around the middle of the title image for best accuracy
    current_height = name_h - 2 - top_margin
    current_width = w - symbol_margin - left_margin
    
    # Apply reductions - centered around middle
    height_reduction = int(current_height * 0.3)  # 30% height reduction
    width_reduction = int(current_height * 0.017)  # 1.7% of height as width reduction
    
    # Center the height reduction (remove from both top and bottom)
    refined_top = top_margin + height_reduction // 2  # Center the height reduction
    refined_bottom = name_h - 2 - (height_reduction - height_reduction // 2)  # Distribute remainder
    
    # Center the width reduction (remove from both left and right)
    refined_left = left_margin + width_reduction // 2  # Center the width reduction
    refined_right = w - symbol_margin - (width_reduction - width_reduction // 2)  # Distribute remainder
    
    # ADDITIONAL: Remove further 1.7% width from RIGHT EDGE only
    additional_right_reduction = int(current_width * 0.017)  # 1.7% of current width
    refined_right = refined_right - additional_right_reduction
    
    print(f"[ocr_description_zone_card_info] Additional right reduction: {additional_right_reduction}px")
    
    name_region = desc_zone_img[refined_top:refined_bottom, refined_left:refined_right]
    
    print(f"[ocr_description_zone_card_info] Applied refinements: height reduced by {height_reduction}px, width reduced by {width_reduction}px")
    
    # Save the specific text region being used for OCR for debugging
    if card_number > 0:
        title_path = Path(f"test_identifier/row{row_number}/title_{card_number:02d}.png")
        title_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(title_path), name_region)
        print(f"[ocr_description_zone_card_info] Saved title region for card {card_number} as {title_path}")
    
    print(f"[ocr_description_zone_card_info] Name region: {name_region.shape} (excluding {symbol_margin}px from right for attribute symbol)")
    
    if name_region.size == 0:
        card_name = ""
    else:
        name_gray = cv2.cvtColor(name_region, cv2.COLOR_BGR2GRAY)
        name_gray = cv2.resize(name_gray, (name_gray.shape[1]*3, name_gray.shape[0]*3), interpolation=cv2.INTER_CUBIC)
        # Use multiple preprocessing approaches for better OCR
        _, name_thresh = cv2.threshold(name_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        name_adaptive = cv2.adaptiveThreshold(name_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Try both preprocessing approaches and pick the one with more text
        config = r'--oem 3 --psm 7'
        try:
            name1 = pytesseract.image_to_string(name_thresh, lang=lang, config=config).strip()
            name2 = pytesseract.image_to_string(name_adaptive, lang=lang, config=config).strip()
            card_name = name1 if len(name1) > len(name2) else name2
            card_name = card_name.replace("\n", " ").replace("\x0c", "").strip()
        except Exception:
            card_name = ""
    
    # NEW: Extract count using count_header.PNG template matching
    count = 1
    
    # Load count_header template
    count_header_path = Path("templates/count_header.PNG")
    if not count_header_path.exists():
        print(f"[ocr_description_zone_card_info] Count header template not found at {count_header_path}")
        return card_name, count
    
    count_header_template = cv2.imread(str(count_header_path))
    if count_header_template is None:
        print(f"[ocr_description_zone_card_info] Failed to load count header template")
        return card_name, count
    
    # Convert to grayscale for template matching
    gray_desc_zone = cv2.cvtColor(desc_zone_img, cv2.COLOR_BGR2GRAY)
    gray_count_header = cv2.cvtColor(count_header_template, cv2.COLOR_BGR2GRAY)
    
    # Perform template matching to find count header
    res = cv2.matchTemplate(gray_desc_zone, gray_count_header, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    
    # Check if match confidence is sufficient
    confidence_threshold = 0.6  # Threshold for count header detection
    if max_val >= confidence_threshold:
        # Get count header position and dimensions
        header_x, header_y = max_loc
        header_h, header_w = gray_count_header.shape[:2]
        print(f"[ocr_description_zone_card_info] Found count header at ({header_x}, {header_y}) with confidence {max_val:.3f}")
        
        # Define count region UNDERNEATH the count header
        count_region_x = header_x
        count_region_y = header_y + header_h + 2  # Start just below the header
        initial_count_region_w = header_w + 20  # Slightly wider than header for count text
        
        # REFINED: Remove 43% width from RIGHT SIDE ONLY to focus on count number
        width_reduction = int(initial_count_region_w * 0.43)  # 43% width reduction from right
        count_region_w = initial_count_region_w - width_reduction
        count_region_h = min(30, h - count_region_y)  # Up to 30px high for count text
        
        print(f"[ocr_description_zone_card_info] Count region refined: removed {width_reduction}px from right (43% reduction)")
        
        # Ensure we don't go outside description zone boundaries
        if count_region_y + count_region_h <= h and count_region_x + count_region_w <= w:
            count_region = desc_zone_img[count_region_y:count_region_y + count_region_h,
                                       count_region_x:count_region_x + count_region_w]
            
            # Save count region for debugging
            if card_number > 0:
                count_path = Path(f"test_identifier/row{row_number}/count_{card_number:02d}.png")
                count_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(count_path), count_region)
                print(f"[ocr_description_zone_card_info] Saved count region (under header) for card {card_number} as {count_path}")
            
            if count_region.size > 0:
                count_gray = cv2.cvtColor(count_region, cv2.COLOR_BGR2GRAY)
                count_gray = cv2.resize(count_gray, (count_gray.shape[1]*3, count_gray.shape[0]*3), interpolation=cv2.INTER_CUBIC)
                
                # Look for numbers using multiple approaches
                _, count_thresh1 = cv2.threshold(count_gray, 120, 255, cv2.THRESH_BINARY_INV)
                _, count_thresh2 = cv2.threshold(count_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                cfg_count = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789x'
                
                for thresh_img in [count_thresh1, count_thresh2]:
                    try:
                        txt = pytesseract.image_to_string(thresh_img, config=cfg_count).strip()
                        # Look for numbers, handle "x3" format
                        if 'x' in txt.lower():
                            num_part = txt.lower().split('x')[-1]
                            if num_part.isdigit():
                                count = int(num_part)
                                break
                        elif txt.isdigit():
                            count = int(txt)
                            break
                    except Exception:
                        continue
        else:
            print(f"[ocr_description_zone_card_info] Count region would exceed description zone boundaries")
    else:
        print(f"[ocr_description_zone_card_info] Count header not found (confidence: {max_val:.3f} < {confidence_threshold})")
    
    print(f"[ocr_description_zone_card_info] Extracted: '{card_name}', count: {count}")
    return card_name, count

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
    # Use proportional corner region for OCR
    corner_h = int(h * 0.4)  # 40% of thumbnail height
    corner_w = int(w * 0.4)  # 40% of thumbnail width
    margin_h = int(h * 0.07)  # 7% margin from bottom
    margin_w = int(w * 0.07)  # 7% margin from right
    cr = thumb_img[max(0, h-corner_h):h-margin_h, max(0, w-corner_w):w-margin_w].copy()
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

# ---------- New Function: Extract First Row of Cards ----------

def detect_card_borders(card_region_img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect the precise borders of a card within a region image.
    Handles normal, glossy, prismatic borders, and glow animations.
    Uses multiple robust techniques to find the card's rectangular border.
    Returns the bounding box (x, y, w, h) relative to card_region_img that contains just the card,
    or None if borders cannot be detected.
    """
    if card_region_img.size == 0:
        return None
    
    h, w = card_region_img.shape[:2]
    region_area = w * h
    
    # Convert to grayscale if needed
    if len(card_region_img.shape) == 3:
        gray = cv2.cvtColor(card_region_img, cv2.COLOR_BGR2GRAY)
        bgr = card_region_img.copy()
    else:
        gray = card_region_img.copy()
        bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    all_candidates = []
    
    # Method 1: Gradient magnitude-based edge detection (works well with glossy/prismatic borders)
    # Compute gradients in both directions
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    max_mag = np.max(magnitude)
    if max_mag > 0:
        magnitude = np.uint8(255 * magnitude / max_mag)
    else:
        magnitude = np.zeros_like(gray, dtype=np.uint8)
    
    # Threshold gradient magnitude to find strong edges (borders)
    for thresh_val in [30, 50, 70, 100]:
        _, edges = cv2.threshold(magnitude, thresh_val, 255, cv2.THRESH_BINARY)
        # Close gaps in borders
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=3)
        # Dilate slightly to connect nearby edges
        dilated = cv2.dilate(closed, kernel, iterations=1)
        
        contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        
        for i, cnt in enumerate(contours):
            # Skip inner contours (holes), focus on outer borders
            if hierarchy is not None and hierarchy[0][i][3] != -1:
                continue
            
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            contour_area = cv2.contourArea(cnt)
            
            if area < 0.20 * region_area or area > 0.98 * region_area:
                continue
            
            aspect = cw / ch if ch > 0 else 0
            if not (0.35 < aspect < 2.8):
                continue
            
            # Calculate how rectangular the contour is
            rect_area = cv2.contourArea(cnt)
            if rect_area > 0:
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                if hull_area > 0:
                    extent = rect_area / hull_area
                    if extent > 0.6:  # Reasonably rectangular
                        all_candidates.append((x, y, cw, ch, area, extent, 'gradient'))
    
    # Method 2: Multi-scale Canny edge detection (handles varying border brightness)
    for blur_size in [(3, 3), (5, 5), (7, 7)]:
        blurred = cv2.GaussianBlur(gray, blur_size, 0)
        # Try multiple Canny thresholds to catch different border types
        for low, high in [(20, 60), (40, 100), (60, 150), (80, 200), (100, 250)]:
            edges = cv2.Canny(blurred, low, high)
            # Strong morphological closing to connect glow effects
            kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            kernel_med = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_large, iterations=2)
            closed = cv2.dilate(closed, kernel_med, iterations=1)
            
            contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            
            for i, cnt in enumerate(contours):
                if hierarchy is not None and hierarchy[0][i][3] != -1:
                    continue
                
                x, y, cw, ch = cv2.boundingRect(cnt)
                area = cw * ch
                contour_area = cv2.contourArea(cnt)
                
                if area < 0.20 * region_area or area > 0.98 * region_area:
                    continue
                
                aspect = cw / ch if ch > 0 else 0
                if not (0.35 < aspect < 2.8):
                    continue
                
                # Check rectangularity
                epsilon = 0.01 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                rect_area = cv2.contourArea(cnt)
                if rect_area > 0:
                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)
                    if hull_area > 0:
                        extent = rect_area / hull_area
                        if extent > 0.5:
                            # Prefer contours closer to 4 vertices (rectangle)
                            vertex_score = 1.0 / (1.0 + abs(len(approx) - 4))
                            all_candidates.append((x, y, cw, ch, area, extent * vertex_score, 'canny'))
    
    # Method 3: Color-based detection (handles prismatic/glossy borders with distinct colors)
    if len(card_region_img.shape) == 3:
        # Try multiple color spaces
        color_spaces = [
            ('BGR', bgr),
            ('HSV', cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)),
            ('LAB', cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)),
        ]
        
        for space_name, img_space in color_spaces:
            # Work with each channel separately
            for channel_idx in range(3):
                channel = img_space[:, :, channel_idx]
                
                # Ensure channel is uint8
                if channel.dtype != np.uint8:
                    channel = channel.astype(np.uint8)
                
                # Try multiple thresholding methods
                # Otsu's method
                try:
                    _, thresh1 = cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                except Exception:
                    continue
                
                # Adaptive threshold (only works if image is large enough)
                try:
                    if channel.shape[0] > 11 and channel.shape[1] > 11:
                        thresh2 = cv2.adaptiveThreshold(channel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                         cv2.THRESH_BINARY, 11, 2)
                    else:
                        thresh2 = thresh1.copy()
                except Exception:
                    thresh2 = thresh1.copy()
                
                for thresh_img in [thresh1, thresh2, cv2.bitwise_not(thresh1), cv2.bitwise_not(thresh2)]:
                    # Clean up the thresholded image
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                    cleaned = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel, iterations=2)
                    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
                    
                    contours, hierarchy = cv2.findContours(cleaned, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    if not contours:
                        continue
                    
                    for i, cnt in enumerate(contours):
                        if hierarchy is not None and hierarchy[0][i][3] != -1:
                            continue
                        
                        x, y, cw, ch = cv2.boundingRect(cnt)
                        area = cw * ch
                        
                        if area < 0.20 * region_area or area > 0.98 * region_area:
                            continue
                        
                        aspect = cw / ch if ch > 0 else 0
                        if not (0.35 < aspect < 2.8):
                            continue
                        
                        contour_area = cv2.contourArea(cnt)
                        if contour_area > 0:
                            hull = cv2.convexHull(cnt)
                            hull_area = cv2.contourArea(hull)
                            if hull_area > 0:
                                extent = contour_area / hull_area
                                if extent > 0.6:
                                    all_candidates.append((x, y, cw, ch, area, extent, f'color_{space_name}'))
    
    # Method 4: HoughLines-based rectangle detection (for straight borders)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=int(min(w, h) * 0.3), 
                            minLineLength=int(min(w, h) * 0.2), maxLineGap=10)
    
    if lines is not None and len(lines) >= 4:
        # Group lines by orientation (horizontal/vertical)
        horizontal_lines = []
        vertical_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 30 or angle > 150:
                horizontal_lines.append((y1 + y2) / 2)
            else:
                vertical_lines.append((x1 + x2) / 2)
        
        if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
            # Find top/bottom and left/right boundaries
            horizontal_lines.sort()
            vertical_lines.sort()
            
            # Take outer boundaries
            top_y = int(min(horizontal_lines[:2]))
            bottom_y = int(max(horizontal_lines[-2:]))
            left_x = int(min(vertical_lines[:2]))
            right_x = int(max(vertical_lines[-2:]))
            
            cw = right_x - left_x
            ch = bottom_y - top_y
            
            if (0.20 * region_area < cw * ch < 0.98 * region_area and 
                0.35 < cw / ch < 2.8 and left_x >= 0 and top_y >= 0 and 
                right_x <= w and bottom_y <= h):
                all_candidates.append((left_x, top_y, cw, ch, cw * ch, 0.9, 'hough'))
    
    # Method 5: Laplacian of Gaussian (LoG) for border detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    
    _, edges_log = cv2.threshold(laplacian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_log = cv2.morphologyEx(edges_log, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    contours, hierarchy = cv2.findContours(closed_log, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        for i, cnt in enumerate(contours):
            if hierarchy is not None and hierarchy[0][i][3] != -1:
                continue
            
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            
            if 0.20 * region_area < area < 0.98 * region_area:
                aspect = cw / ch if ch > 0 else 0
                if 0.35 < aspect < 2.8:
                    contour_area = cv2.contourArea(cnt)
                    if contour_area > 0:
                        hull = cv2.convexHull(cnt)
                        hull_area = cv2.contourArea(hull)
                        if hull_area > 0:
                            extent = contour_area / hull_area
                            if extent > 0.5:
                                all_candidates.append((x, y, cw, ch, area, extent, 'log'))
    
    # Evaluate all candidates and pick the best one
    if all_candidates:
        # Remove duplicates and similar boxes (merge if overlap > 80%)
        unique_candidates = []
        for candidate in all_candidates:
            x, y, cw, ch, area, score, method = candidate
            is_duplicate = False
            
            for ux, uy, uw, uh, uarea, uscore, umethod in unique_candidates:
                # Check overlap
                overlap_x = max(0, min(x + cw, ux + uw) - max(x, ux))
                overlap_y = max(0, min(y + ch, uy + uh) - max(y, uy))
                overlap_area = overlap_x * overlap_y
                union_area = area + uarea - overlap_area
                
                if union_area > 0 and overlap_area / union_area > 0.8:
                    # Keep the one with better score
                    if score > uscore:
                        unique_candidates.remove((ux, uy, uw, uh, uarea, uscore, umethod))
                        unique_candidates.append(candidate)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_candidates.append(candidate)
        
        # Sort by score (rectangularity/extent) and area appropriateness
        # Prefer candidates that are:
        # 1. More rectangular (higher extent score)
        # 2. Reasonably sized (not too small, not too large)
        # 3. Centered in the region
        def candidate_score(cand):
            x, y, cw, ch, area, extent_score, method = cand
            size_score = 1.0 - abs(area / region_area - 0.6)  # Prefer ~60% of region
            center_x, center_y = w / 2, h / 2
            card_center_x, card_center_y = x + cw / 2, y + ch / 2
            center_score = 1.0 - (abs(card_center_x - center_x) / w + abs(card_center_y - center_y) / h) / 2
            return extent_score * 0.5 + size_score * 0.3 + center_score * 0.2
        
        unique_candidates.sort(key=candidate_score, reverse=True)
        
        # Take the best candidate
        x, y, cw, ch, area, score, method_name = unique_candidates[0]
        print(f"[detect_card_borders] Best candidate: method={method_name}, score={score:.3f}, size={cw}x{ch}")
        
        # Add small padding to ensure we capture the full border
        padding = 3
        x = max(0, x - padding)
        y = max(0, y - padding)
        cw = min(w - x, cw + 2 * padding)
        ch = min(h - y, ch + 2 * padding)
        
        return (x, y, cw, ch)
    
    # Fallback Method: Conservative edge-based approach with very loose criteria
    # This is a last resort that should catch most cards even with difficult borders
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 10, 50)  # Very sensitive thresholds
    
    # Large kernel to close any gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=4)
    closed = cv2.dilate(closed, kernel, iterations=2)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the largest contour that's reasonably sized
        candidates = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            
            if area < 0.15 * region_area or area > 0.98 * region_area:
                continue
            
            aspect = cw / ch if ch > 0 else 0
            if 0.3 < aspect < 3.0:  # Very loose aspect ratio
                candidates.append((x, y, cw, ch, area))
        
        if candidates:
            # Sort by area and take the largest reasonable one
            candidates.sort(key=lambda c: c[4], reverse=True)
            x, y, cw, ch, _ = candidates[0]
            
            # Add small padding
            padding = 2
            x = max(0, x - padding)
            y = max(0, y - padding)
            cw = min(w - x, cw + 2 * padding)
            ch = min(h - y, ch + 2 * padding)
            
            print(f"[detect_card_borders] Fallback method found border: size={cw}x{ch}")
            return (x, y, cw, ch)
    
    # If even fallback fails, return None
    print("[detect_card_borders] All detection methods including fallback failed")
    return None

def click_cards_and_extract_info_single_row(win, row_number: int = 1) -> Dict[str, int]:
    """
    CHANGE 2: Process a single row of 6 cards - detects cards, clicks each one,
    captures the description zone, extracts card name and count, and returns summary.
    Returns dictionary with card names as keys and total counts as values.
    """
    print(f"[click_cards_and_extract_info_single_row] Starting row {row_number} card detection and clicking process...")
    
    # Step 1: Get window coordinates and handle negative coords
    left, top, width, height = win.left, win.top, win.width, win.height
    
    if left < 0 or top < 0:
        print(f"[click_cards_and_extract_info] Window at negative coords ({left},{top}). Attempting to move window to primary monitor (8,8).")
        try:
            win.moveTo(8, 8)
            time.sleep(0.35)
            left, top, width, height = win.left, win.top, win.width, win.height
            print(f"[click_cards_and_extract_info] Window moved to {left},{top} (size {width}x{height})")
        except Exception as e:
            print(f"[click_cards_and_extract_info] Could not move window: {e}. Proceeding with original coords.")
    
    # Step 2: Grab full window screenshot
    try:
        full_window_img = grab_region((left, top, width, height))
        print(f"[click_cards_and_extract_info] Captured window screenshot: {full_window_img.shape}")
    except Exception as e:
        print(f"[click_cards_and_extract_info] Failed to grab window region: {e}")
        return {}
    
    # Step 3: Load header template
    header_template_path = Path("templates/header.PNG")
    if not header_template_path.exists():
        print(f"[click_cards_and_extract_info] Header template not found at {header_template_path}")
        return {}
    
    header_template = cv2.imread(str(header_template_path))
    if header_template is None:
        print(f"[click_cards_and_extract_info] Failed to load header template from {header_template_path}")
        return {}
    
    print(f"[click_cards_and_extract_info] Loaded header template: {header_template.shape}")
    
    # Step 4: Find header in window using template matching
    gray_window = cv2.cvtColor(full_window_img, cv2.COLOR_BGR2GRAY)
    gray_header = cv2.cvtColor(header_template, cv2.COLOR_BGR2GRAY)
    
    res = cv2.matchTemplate(gray_window, gray_header, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    
    if max_val < 0.5:
        print(f"[click_cards_and_extract_info] Header template match confidence too low: {max_val:.3f} (threshold: 0.5)")
        return {}
    
    header_x, header_y = max_loc
    header_h, header_w = gray_header.shape[:2]
    print(f"[click_cards_and_extract_info] Found header at window-relative position: ({header_x}, {header_y}) with confidence {max_val:.3f}")
    
    # Step 5: SIMPLIFIED - Use same fixed detection area for all rows
    # After scrolling, the next row should appear in the same relative position
    
    # Card area horizontal boundaries (consistent for all rows)
    card_area_margin = int(header_w * 0.02)  # 2% margin from header edge to first card
    card_area_x = header_x + card_area_margin
    # EXPANDED: Increase width to capture full 6th card (was 90%, now increased to ~93.5% to fix 96.5% issue)
    card_area_w = int(header_w * 0.935)  # Cards use ~93.5% of header width to include full 6th card
    
    # Use fixed positioning for all rows - after scroll, next row appears in same position
    card_area_y = header_y + header_h + 10
    
    # Estimated card dimensions
    estimated_card_width = card_area_w // 6
    estimated_card_height = int(estimated_card_width * 1.4)
    card_area_h = estimated_card_height + 20
    
    # Ensure we don't go outside window boundaries
    card_area_h = min(card_area_h, height - card_area_y)
    
    print(f"[click_cards_and_extract_info_single_row] FIXED position for row {row_number}: x={card_area_x}, y={card_area_y}, w={card_area_w}, h={card_area_h}")
    
    if card_area_w <= 0 or card_area_h <= 0:
        print(f"[click_cards_and_extract_info] Card area dimensions invalid: {card_area_w}x{card_area_h}")
        return {}
    
    print(f"[click_cards_and_extract_info] Card collection area: {card_area_x}, {card_area_y}, {card_area_w}x{card_area_h}")
    
    # Step 6: FIXED CARD DETECTION - Use reference image approach for reliable card positions
    # Save the full window screenshot for debugging
    cv2.imwrite("tmp_rovodev_full_window.png", full_window_img)
    print(f"[click_cards_and_extract_info] Saved full window screenshot as tmp_rovodev_full_window.png")
    
    # Extract card area based on reference image dimensions
    card_area_img = full_window_img[card_area_y:card_area_y + card_area_h, card_area_x:card_area_x + card_area_w]
    cv2.imwrite("tmp_rovodev_card_area.png", card_area_img)
    print(f"[click_cards_and_extract_info] Saved card area as tmp_rovodev_card_area.png")
    
    # NEW: Save row_full screenshot with card boundaries like first_row_full.png
    row_full_img = card_area_img.copy()
    
    # Draw card boundary lines on the row_full image for visualization
    card_width = card_area_w // 6
    for i in range(1, 6):  # Draw 5 vertical lines to separate 6 cards
        line_x = i * card_width
        cv2.line(row_full_img, (line_x, 0), (line_x, card_area_h), (0, 255, 0), 2)  # Green lines
    
    # Save the row_full screenshot with boundaries
    row_full_path = Path(f"test_identifier/row{row_number}/{row_number}_row_full.png")
    row_full_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(row_full_path), row_full_img)
    print(f"[click_cards_and_extract_info_single_row] Saved row {row_number} full screenshot with boundaries as {row_full_path}")
    
    # IMPROVED: Use fixed grid approach based on reference image
    # From first_row_full.png, we can see 6 cards in a horizontal row
    # Each card should be approximately equal width
    area_h, area_w = card_area_img.shape[:2]
    
    # Calculate card positions using equal-width division
    card_width = area_w // 6  # Divide area width by 6 cards
    card_height = min(area_h, int(card_width * 1.4))  # Card aspect ratio approximately 1:1.4
    
    # Start from top of area, center cards vertically if needed
    start_y = max(0, (area_h - card_height) // 2)
    
    cards_to_process = []
    
    for i in range(6):
        # Calculate card position
        card_x = i * card_width
        card_y = start_y
        
        # Add small margins to ensure we don't cut card borders
        margin_x = int(card_width * 0.05)  # 5% margin
        margin_y = int(card_height * 0.05)
        
        # Adjust boundaries to include margins but stay within bounds
        final_x = max(0, card_x + margin_x)
        final_y = max(0, card_y + margin_y) 
        final_w = min(area_w - final_x, card_width - 2 * margin_x)
        final_h = min(area_h - final_y, card_height - 2 * margin_y)
        
        if final_w > 10 and final_h > 10:  # Ensure minimum size
            # Extract card image
            card_img = card_area_img[final_y:final_y + final_h, final_x:final_x + final_w].copy()
            
            # Save individual card for debugging in row-specific folder
            card_path = Path(f"test_identifier/row{row_number}/card_{i+1:02d}.png")
            card_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(card_path), card_img)
            print(f"[click_cards_and_extract_info_single_row] Saved detected card {i+1} as {card_path}")
            
            # Store card with its bounding box (relative to card area)
            cards_to_process.append((card_img, (final_x, final_y, final_w, final_h)))
            print(f"[click_cards_and_extract_info] Card {i+1}: position=({final_x},{final_y}), size=({final_w}x{final_h})")
        else:
            print(f"[click_cards_and_extract_info] Warning: Card {i+1} has invalid dimensions: {final_w}x{final_h}")
    
    print(f"[click_cards_and_extract_info] Successfully detected {len(cards_to_process)} cards using fixed grid approach")
    
    # Step 7: Enhanced description zone detection function
    def detect_and_capture_description_zone(window_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect the card_header.PNG template on screen and capture the description area below it.
        Returns the description zone image or None if failed.
        """
        # Load card_header template
        card_header_path = Path("templates/card_header.PNG")
        if not card_header_path.exists():
            print(f"[detect_and_capture_description_zone] Card header template not found at {card_header_path}")
            return None
        
        card_header_template = cv2.imread(str(card_header_path))
        if card_header_template is None:
            print(f"[detect_and_capture_description_zone] Failed to load card header template")
            return None
        
        # Convert to grayscale for template matching
        gray_window = cv2.cvtColor(window_img, cv2.COLOR_BGR2GRAY)
        gray_card_header = cv2.cvtColor(card_header_template, cv2.COLOR_BGR2GRAY)
        
        # Perform template matching
        res = cv2.matchTemplate(gray_window, gray_card_header, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # Check if match confidence is sufficient
        confidence_threshold = 0.4  # Lower threshold for more flexibility
        if max_val < confidence_threshold:
            print(f"[detect_and_capture_description_zone] Card header template match confidence too low: {max_val:.3f} (threshold: {confidence_threshold})")
            return None
        
        # Get header position and dimensions
        header_x, header_y = max_loc
        header_h, header_w = gray_card_header.shape[:2]
        print(f"[detect_and_capture_description_zone] Found card header at position: ({header_x}, {header_y}) with confidence {max_val:.3f}")
        
        # Define description zone area below the header
        desc_zone_x = header_x
        desc_zone_y = header_y + header_h + 5  # Small gap after header
        # Use proportional dimensions based on window size
        window_h, window_w = window_img.shape[:2]
        desc_zone_w = min(int(window_w * 0.22), header_w * 3)  # ~22% of window width for card info box
        
        # CHANGE 1: Expand description zone height by 73% from the original 16%
        original_desc_zone_h = int(window_h * 0.16)  # Original 16% of window height
        desc_zone_h = int(original_desc_zone_h * 1.73)  # Increase by 73% for expanded capture area
        print(f"[detect_and_capture_description_zone] Expanded description zone height from {original_desc_zone_h} to {desc_zone_h} pixels (73% increase)")
        
        # Ensure we don't go outside window boundaries
        desc_zone_h = min(desc_zone_h, window_h - desc_zone_y)
        
        if desc_zone_w <= 0 or desc_zone_h <= 0:
            print(f"[detect_and_capture_description_zone] Description zone dimensions invalid: {desc_zone_w}x{desc_zone_h}")
            return None
        
        # REFINED: Trim description zone - remove 2% height from bottom, 4% width from right
        height_reduction = int(desc_zone_h * 0.02)  # 2% height reduction from bottom
        width_reduction = int(desc_zone_w * 0.04)   # 4% width reduction from right
        
        refined_desc_zone_h = desc_zone_h - height_reduction
        refined_desc_zone_w = desc_zone_w - width_reduction
        
        print(f"[detect_and_capture_description_zone] Trimmed description zone: removed {height_reduction}px from bottom, {width_reduction}px from right")
        
        # Capture the refined description zone
        description_zone = window_img[desc_zone_y:desc_zone_y + refined_desc_zone_h,
                                    desc_zone_x:desc_zone_x + refined_desc_zone_w]
        
        print(f"[detect_and_capture_description_zone] Successfully captured refined description zone: {refined_desc_zone_w}x{refined_desc_zone_h} pixels (original: {desc_zone_w}x{desc_zone_h})")
        return description_zone
    
    # Step 8: Process each card by clicking and extracting info
    card_summary = {}  # Dictionary to store card name -> total count
    
    for i, (card_img, card_bbox) in enumerate(cards_to_process):
        print(f"\n[click_cards_and_extract_info] Processing card {i+1}/{len(cards_to_process)}")
        
        # Calculate click position (center of card relative to screen)
        # card_bbox is relative to card_area_img, so we need to add offsets
        card_x, card_y, card_w, card_h = card_bbox
        
        # FIXED: Proper click position calculation
        # Add window position + card_area position + card position within area
        click_x = left + card_area_x + card_x + card_w // 2
        click_y = top + card_area_y + card_y + card_h // 2
        
        print(f"[click_cards_and_extract_info] Card {i+1} calculation:")
        print(f"  Window: ({left}, {top})")
        print(f"  Card area offset: ({card_area_x}, {card_area_y})")  
        print(f"  Card bbox: ({card_x}, {card_y}, {card_w}, {card_h})")
        print(f"  Final click position: ({click_x}, {click_y})")
        
        try:
            # Click on the card
            pyautogui.click(click_x, click_y)
            time.sleep(0.8)  # Wait for description to appear
            
            # Capture new screenshot after clicking
            clicked_window_img = grab_region((left, top, width, height))
            
            # Detect and capture description zone
            desc_zone_img = detect_and_capture_description_zone(clicked_window_img)
            
            if desc_zone_img is not None:
                # Save description zone in row-specific folder
                desc_path = Path(f"test_identifier/row{row_number}/desc_{i+1:02d}.png")
                desc_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(desc_path), desc_zone_img)
                print(f"[click_cards_and_extract_info_single_row] Saved description zone {i+1} as {desc_path}")
                
                # Extract card name and count from description zone
                card_name, count = ocr_description_zone_card_info(desc_zone_img, card_number=i+1, row_number=row_number)
                
                if card_name:  # Only process if we got a card name
                    # CHANGE 3: Aggregate counts for duplicate card names
                    if card_name in card_summary:
                        card_summary[card_name] += count
                        print(f"[click_cards_and_extract_info] Updated existing card '{card_name}': now {card_summary[card_name]} total")
                    else:
                        card_summary[card_name] = count
                        print(f"[click_cards_and_extract_info] New card '{card_name}': {count}")
                else:
                    print(f"[click_cards_and_extract_info] Could not extract card name from description zone")
            else:
                print(f"[click_cards_and_extract_info] Could not detect description zone for card {i+1}")
                
        except Exception as e:
            print(f"[click_cards_and_extract_info] Error processing card {i+1}: {e}")
            continue
    
    print(f"\n[click_cards_and_extract_info_single_row] Completed processing {len(cards_to_process)} cards for row {row_number}")
    return card_summary

def click_cards_and_extract_info_multi_row(win, max_rows: int = 2) -> List[Tuple[str, int]]:
    """
    EXPANDED: Process multiple rows of cards by detecting each row and clicking all cards.
    Currently processes up to 2 rows as requested.
    Returns list of tuples (card_name, count) in encounter order.
    """
    print(f"[click_cards_and_extract_info_multi_row] Starting multi-row processing for {max_rows} rows...")
    
    # Use ordered list to maintain encounter order instead of dictionary
    cards_in_order = []
    
    for row_num in range(1, max_rows + 1):
        print(f"\n{'='*50}")
        print(f"PROCESSING ROW {row_num}")
        print(f"{'='*50}")
        
        # Process current row
        row_summary = click_cards_and_extract_info_single_row(win, row_number=row_num)
        
        # Add results from this row in encounter order
        for card_name, count in row_summary.items():
            # Skip empty card names
            if not card_name or card_name.strip() == "":
                print(f"[multi_row] Skipping empty card name")
                continue
                
            # Check if card already encountered
            found_existing = False
            for i, (existing_name, existing_count) in enumerate(cards_in_order):
                if existing_name == card_name:
                    # Update existing entry
                    cards_in_order[i] = (existing_name, existing_count + count)
                    print(f"[multi_row] Combined counts for '{card_name}': now {existing_count + count} total")
                    found_existing = True
                    break
            
            if not found_existing:
                # Add new card in encounter order
                cards_in_order.append((card_name, count))
                print(f"[multi_row] New card '{card_name}': {count}")
        
        print(f"[multi_row] Row {row_num} completed. Current total unique cards: {len(cards_in_order)}")
        
        # If this is not the last row, scroll down to reveal next row
        if row_num < max_rows:
            print(f"[multi_row] Scrolling down to reveal row {row_num + 1}...")
            
            # REDUCED: Use smaller, more controlled scroll to avoid overshooting
            # Single small scroll to shift down to next row
            pyautogui.scroll(-1)  # Very small scroll down
            time.sleep(1.5)  # Longer wait for UI to stabilize
            print(f"[multi_row] Gentle scroll completed, ready for row {row_num + 1}")
    
    print(f"\n[click_cards_and_extract_info_multi_row] Completed processing {max_rows} rows")
    return cards_in_order

def find_and_extract_first_row_cards(win) -> bool:
    """
    Find the header template, locate the card collection area below it,
    detect the first row of 6 cards, and save each card thumbnail to test_identifier folder.
    Returns True if successful, False otherwise.
    """
    print("[find_and_extract_first_row_cards] Starting card detection...")
    
    # Step 1: Get window coordinates and handle negative coords
    left, top, width, height = win.left, win.top, win.width, win.height
    
    if left < 0 or top < 0:
        print(f"[find_and_extract_first_row_cards] Window at negative coords ({left},{top}). Attempting to move window to primary monitor (8,8).")
        try:
            win.moveTo(8, 8)
            time.sleep(0.35)
            left, top, width, height = win.left, win.top, win.width, win.height
            print(f"[find_and_extract_first_row_cards] Window moved to {left},{top} (size {width}x{height})")
        except Exception as e:
            print(f"[find_and_extract_first_row_cards] Could not move window: {e}. Proceeding with original coords.")
    
    # Step 2: Grab full window screenshot
    try:
        full_window_img = grab_region((left, top, width, height))
        print(f"[find_and_extract_first_row_cards] Captured window screenshot: {full_window_img.shape}")
    except Exception as e:
        print(f"[find_and_extract_first_row_cards] Failed to grab window region: {e}")
        return False
    
    # Step 3: Load header template
    header_template_path = Path("templates/header.PNG")
    if not header_template_path.exists():
        print(f"[find_and_extract_first_row_cards] Header template not found at {header_template_path}")
        return False
    
    header_template = cv2.imread(str(header_template_path))
    if header_template is None:
        print(f"[find_and_extract_first_row_cards] Failed to load header template from {header_template_path}")
        return False
    
    print(f"[find_and_extract_first_row_cards] Loaded header template: {header_template.shape}")
    
    # Step 4: Find header in window using template matching
    gray_window = cv2.cvtColor(full_window_img, cv2.COLOR_BGR2GRAY)
    gray_header = cv2.cvtColor(header_template, cv2.COLOR_BGR2GRAY)
    
    res = cv2.matchTemplate(gray_window, gray_header, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    
    if max_val < 0.5:  # Lowered for better detection with proportional changes
        print(f"[find_and_extract_first_row_cards] Header template match confidence too low: {max_val:.3f} (threshold: 0.65)")
        return False
    
    header_x, header_y = max_loc
    _, header_h = gray_header.shape[1], gray_header.shape[0]
    print(f"[find_and_extract_first_row_cards] Found header at window-relative position: ({header_x}, {header_y}) with confidence {max_val:.3f}")
    
    # Define description zone detection function first
    def detect_and_capture_description_zone(window_img: np.ndarray) -> bool:
        """
        Detect the card_header.PNG template on screen and capture the description area below it.
        This area shows card titles and information when a card is clicked.
        Returns True if successfully captured, False otherwise.
        """
        # Load card_header template
        card_header_path = Path("templates/card_header.PNG")
        if not card_header_path.exists():
            print(f"[detect_and_capture_description_zone] Card header template not found at {card_header_path}")
            return False
        
        card_header_template = cv2.imread(str(card_header_path))
        if card_header_template is None:
            print(f"[detect_and_capture_description_zone] Failed to load card header template")
            return False
        
        print(f"[detect_and_capture_description_zone] Loaded card header template: {card_header_template.shape}")
        
        # Convert to grayscale for template matching
        gray_window = cv2.cvtColor(window_img, cv2.COLOR_BGR2GRAY)
        gray_card_header = cv2.cvtColor(card_header_template, cv2.COLOR_BGR2GRAY)
        
        # Perform template matching
        res = cv2.matchTemplate(gray_window, gray_card_header, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # Check if match confidence is sufficient
        confidence_threshold = 0.5  # Lowered for better detection with proportional changes
        if max_val < confidence_threshold:
            print(f"[detect_and_capture_description_zone] Card header template match confidence too low: {max_val:.3f} (threshold: {confidence_threshold})")
            return False
        
        # Get header position and dimensions
        header_x, header_y = max_loc
        header_h, header_w = gray_card_header.shape[:2]
        print(f"[detect_and_capture_description_zone] Found card header at position: ({header_x}, {header_y}) with confidence {max_val:.3f}")
        
        # Define description zone area below the header
        # Only capture the leftmost card description box, not the deck area
        desc_zone_x = header_x
        desc_zone_y = header_y + header_h + 5  # Small gap after header
        # Use proportional dimensions based on window size
        window_h, window_w = window_img.shape[:2]
        desc_zone_w = min(int(window_w * 0.22), header_w * 3)  # ~22% of window width for card info box
        # CHANGE 1: Expand description zone height by 73% from the original 16%
        original_desc_zone_h = int(window_h * 0.16)  # Original 16% of window height
        desc_zone_h = int(original_desc_zone_h * 1.73)  # Increase by 73% for expanded capture area
        print(f"[detect_and_capture_description_zone] Expanded description zone height from {original_desc_zone_h} to {desc_zone_h} pixels (73% increase)")
        
        # Ensure we don't go outside window boundaries
        window_h, window_w = window_img.shape[:2]
        desc_zone_w = min(desc_zone_w, window_w - desc_zone_x)
        desc_zone_h = min(desc_zone_h, window_h - desc_zone_y)
        
        if desc_zone_w <= 0 or desc_zone_h <= 0:
            print(f"[detect_and_capture_description_zone] Description zone dimensions invalid: {desc_zone_w}x{desc_zone_h}")
            return False
        
        # Extract description zone
        description_zone = window_img[desc_zone_y:desc_zone_y + desc_zone_h, 
                                    desc_zone_x:desc_zone_x + desc_zone_w].copy()
        
        # Save description zone image
        output_path = Path("test_identifier/description_zone.png")
        output_path.parent.mkdir(exist_ok=True)
        
        success = cv2.imwrite(str(output_path), description_zone)
        if success:
            print(f"[detect_and_capture_description_zone] Successfully saved description zone: {desc_zone_w}x{desc_zone_h} pixels to {output_path}")
            return True
        else:
            print(f"[detect_and_capture_description_zone] Failed to save description zone image")
            return False
    
    # Step 4.5: Detect and capture description zone (card info area)
    description_captured = detect_and_capture_description_zone(full_window_img)
    if description_captured:
        print(f"[find_and_extract_first_row_cards] Description zone captured successfully")
    else:
        print(f"[find_and_extract_first_row_cards] Warning: Could not capture description zone")
    
    # Step 5: Define card collection area below header
    # Collection area starts slightly below the header template
    window_h, window_w = full_window_img.shape[:2]
    collection_start_y = header_y + header_h + int(window_h * 0.01)  # 1% of window height gap after header
    collection_end_y = height  # Go to bottom of window
    collection_start_x = header_x  # Start at header's left edge
    collection_end_x = width  # Go to right edge of window
    
    # Extract collection area region
    collection_region_img = full_window_img[collection_start_y:collection_end_y, collection_start_x:collection_end_x]
    
    if collection_region_img.size == 0:
        print("[find_and_extract_first_row_cards] Collection region is empty")
        return False
    
    print(f"[find_and_extract_first_row_cards] Collection region: {collection_region_img.shape} (window-relative: x={collection_start_x}, y={collection_start_y})")
    
    # Step 6: First, detect the right edge of the card collection area by finding the slider background
    def detect_collection_right_edge(img: np.ndarray) -> int:
        """
        Simple and effective detection of where card collection ends before slider area.
        Uses brightness analysis to find dark slider area and subtracts a buffer to exclude
        the gray area completely.
        """
        if len(img.shape) != 3:
            return img.shape[1]

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate column-wise average brightness
        col_brightness = gray.mean(axis=0)
        
        # Calculate baseline brightness from left portion (card collection area)
        baseline_sample_width = min(int(w * 0.2), w // 4)  # 20% of width or 1/4, whichever is smaller
        baseline_brightness = col_brightness[:baseline_sample_width].mean()
        
        print(f"[detect_collection_right_edge] Baseline brightness: {baseline_brightness:.1f}")
        
        # Look for significant brightness drop (dark slider area)
        darkness_threshold = baseline_brightness * 0.65  # Slider is much darker
        scan_width = min(int(w * 0.4), w // 3)  # 40% of width or 1/3, whichever is smaller
        
        for x in range(w - int(w * 0.02), max(0, w - scan_width), -1):  # Start 2% from right edge
            if col_brightness[x] < darkness_threshold:
                # Found dark area - this is likely the slider
                # Move boundary left by a percentage of width to exclude ALL gray slider background
                buffer_pixels = int(w * 0.12)  # 12% of width buffer to eliminate slider area
                boundary = max(0, x - buffer_pixels)
                print(f"[detect_collection_right_edge] Found dark area at x={x}, boundary at x={boundary} ({buffer_pixels}px buffer, 12% of width)")
                return boundary
        
        # Fallback: Use 89% of width
        print(f"[detect_collection_right_edge] No dark area found, triggering 89% fallback")
        return None
    
    # Detect the actual collection width (excluding slider)
    collection_right_edge = detect_collection_right_edge(collection_region_img)
    
    # Handle fallback case where slider detection failed
    use_fallback_crop = False
    if collection_right_edge is None:
        use_fallback_crop = True
        # Use full width for detection, but crop to 89% for final output
        collection_right_edge = collection_region_img.shape[1]  # Full width for now
        print(f"[find_and_extract_first_row_cards] Slider detection failed, using 89% width fallback")
    else:
        print(f"[find_and_extract_first_row_cards] Detected collection right edge at x={collection_right_edge}")
    
    # Detect the left border (grey area) to exclude it from the row image
    def detect_collection_left_edge(img: np.ndarray) -> int:
        """
        Detect where the thin left border ends and the card collection area begins.
        The border is typically a thin dark line or slightly different colored area.
        """
        if len(img.shape) != 3:
            return 0
        
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate column-wise brightness to find transition
        col_brightness = gray.mean(axis=0)
        
        # Look for baseline collection brightness from further right
        baseline_start = min(int(w * 0.12), w // 4)  # 12% of width or 1/4, whichever is smaller
        baseline_end = min(int(w * 0.25), w // 3)  # 25% of width or 1/3, whichever is smaller
        if baseline_end > baseline_start and baseline_end < w:
            baseline_brightness = col_brightness[baseline_start:baseline_end].mean()
        else:
            baseline_brightness = col_brightness[min(int(w * 0.06), w//2):].mean()  # 6% of width or half, whichever is smaller
        
        print(f"[detect_collection_left_edge] Baseline brightness: {baseline_brightness:.1f}")
        
        # Look for thin border (proportional to screen width)
        max_border_width = min(int(w * 0.03), w // 8)  # 3% of width or 1/8, whichever is smaller
        brightness_threshold = baseline_brightness * 0.9  # 90% of collection brightness
        
        # Find first position where brightness consistently matches collection area
        for x in range(max_border_width):
            if x >= w:
                break
            
            # Check current position and a few pixels ahead for consistency
            check_width = min(int(w * 0.01), w - x)  # 1% of width for consistency check
            if check_width > 0:
                avg_brightness = col_brightness[x:x+check_width].mean()
                
                # If this position and nearby are bright enough, this is likely collection start
                if avg_brightness >= brightness_threshold:
                    # Verify consistency - check that it doesn't drop significantly in next few pixels
                    consistent = True
                    extended_check = min(int(w * 0.02), w - x)  # 2% of width for extended check
                    if extended_check > check_width:
                        extended_brightness = col_brightness[x:x+extended_check].mean()
                        if extended_brightness < brightness_threshold * 0.85:
                            consistent = False
                    
                    if consistent:
                        print(f"[detect_collection_left_edge] Found collection start at x={x}")
                        return x
        
        # Fallback: skip a small border width
        fallback_edge = min(int(w * 0.006), w // 30)  # 0.6% of width or 1/30, whichever is smaller
        print(f"[detect_collection_left_edge] Using fallback left edge at x={fallback_edge}")
        return fallback_edge
    
    collection_left_edge = detect_collection_left_edge(collection_region_img)
    print(f"[find_and_extract_first_row_cards] Detected collection left edge at x={collection_left_edge}")
    
    # Step 7: Detect first row of 6 cards
    # Convert to grayscale for contour detection
    gray_collection = cv2.cvtColor(collection_region_img, cv2.COLOR_BGR2GRAY)
    h_collection, w_collection = gray_collection.shape
    
    # Use the detected edges as the effective collection area for card detection
    w_collection_effective = collection_right_edge - collection_left_edge
    x_collection_offset = collection_left_edge
    
    # Method 1: Try contour detection
    blurred = cv2.GaussianBlur(gray_collection, (3, 3), 0)
    edges = cv2.Canny(blurred, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    card_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        
        # Filter by area and aspect ratio
        if area < MIN_CARD_AREA:
            continue
        if area > MAX_CARD_AREA_RATIO * (w_collection * h_collection):
            continue
        aspect = w / h if h > 0 else 0
        if not (CARD_ASPECT_MIN < aspect < CARD_ASPECT_MAX):
            continue
        
        card_boxes.append((x, y, w, h, area))
    
    # Process detected boxes
    if len(card_boxes) >= 6:
        # Remove overlapping boxes, keeping the largest
        card_boxes_sorted = sorted(card_boxes, key=lambda b: b[4], reverse=True)  # Sort by area
        kept_boxes = []
        for b in card_boxes_sorted:
            x, y, w, h, area = b
            overlap = False
            for k in kept_boxes:
                kx, ky, kw, kh, _ = k
                # Check if boxes overlap significantly (more than 30% overlap)
                overlap_x = max(0, min(x + w, kx + kw) - max(x, kx))
                overlap_y = max(0, min(y + h, ky + kh) - max(y, ky))
                overlap_area = overlap_x * overlap_y
                union_area = w * h + kw * kh - overlap_area
                if union_area > 0 and (overlap_area / float(union_area)) > 0.3:
                    overlap = True
                    break
            if not overlap:
                kept_boxes.append(b)
        
        # Sort by y-coordinate (top to bottom), then by x-coordinate (left to right)
        kept_boxes = sorted(kept_boxes, key=lambda b: (b[1], b[0]))
        
        # Take only the first row (cards with similar y-coordinates)
        if len(kept_boxes) > 0:
            first_row_y = kept_boxes[0][1]
            
            # Create a region for the first row based on the detected cards
            first_row_h = max(b[3] for b in kept_boxes)  # maximum height of any card
            first_row_img = collection_region_img[0:first_row_y + first_row_h, :]
            
            # Now we can calculate the tolerance based on the first row height
            y_tolerance = int(first_row_img.shape[0] * 0.26)  # 26% of row height tolerance
            
            first_row_boxes = [b for b in kept_boxes if abs(b[1] - first_row_y) <= y_tolerance]
            # Sort by x-coordinate to get left-to-right order
            first_row_boxes = sorted(first_row_boxes, key=lambda b: b[0])
            # Take first 6 cards
            if len(first_row_boxes) >= 6:
                card_boxes = first_row_boxes[:6]
            else:
                card_boxes = first_row_boxes
                print(f"[find_and_extract_first_row_cards] Only found {len(card_boxes)} cards in first row via contour detection")
    
    # Method 2: If contour detection didn't find 6 cards, use grid-based approach
    if len(card_boxes) < 6:
        print(f"[find_and_extract_first_row_cards] Contour detection found {len(card_boxes)} cards, trying grid-based detection...")
        # Calculate approximate card dimensions based on effective collection width (6 columns)
        card_width = w_collection_effective // 6
        
        # Estimate card height - cards are roughly square, but might be slightly taller
        # Cards are typically square-ish, so start with card_width
        card_height = int(card_width * 1.1)  # Slightly taller than wide
        
        # Create 6 boxes for the first row
        card_boxes = []
        for col in range(6):
            x = collection_left_edge + col * card_width
            y = 0  # First row starts at top of collection region
            w = card_width
            h = min(h_collection, int(card_width * 1.4))  # Use full height available, up to 1.4x width
            
            # Make sure we don't exceed bounds
            if x + w > collection_right_edge:
                w = collection_right_edge - x
            if y + h > h_collection:
                h = h_collection - y
            
            if w > 10 and h > 10:  # Valid dimensions
                card_boxes.append((x, y, w, h, w * h))
        
        print(f"[find_and_extract_first_row_cards] Grid-based detection created {len(card_boxes)} card boxes")
    
    if len(card_boxes) < 6:
        print(f"[find_and_extract_first_row_cards] Could not detect 6 cards. Found {len(card_boxes)} cards.")
        return False
    
    print(f"[find_and_extract_first_row_cards] Detected {len(card_boxes)} cards in first row")
    
    # Step 8: Save a screenshot of the entire first row for visualization
    output_dir = Path("test_identifier")
    output_dir.mkdir(exist_ok=True)
    
    # Calculate the bounding box that encompasses all cards in the first row
    if len(card_boxes) > 0:
        # Find the min/max coordinates of all card boxes
        min_x = min(box[0] for box in card_boxes)
        min_y = min(box[1] for box in card_boxes)
        max_x = max(box[0] + box[2] for box in card_boxes)  # x + width
        max_y = max(box[1] + box[3] for box in card_boxes)  # y + height
        
        # Add some padding around the row for context, but respect the left edge
        row_padding = int(collection_right_edge * 0.035)  # 3.5% of collection width for padding
        row_x = max(collection_left_edge, min_x - row_padding)
        row_y = 0  # Start from top of collection region to capture full height
        
        # Calculate row width based on the rightmost card position plus padding
        # This ensures we only capture the card row itself, not the slider area
        calculated_row_w = max_x - row_x + row_padding
        
        # Use the minimum of calculated width and detected edge to ensure we don't exceed bounds
        max_allowed_w = collection_right_edge - row_x
        row_w = min(calculated_row_w, max_allowed_w, w_collection - row_x)
        # Use full height of the first row area
        row_h = max(max_y + row_padding, int(card_width * 1.4)) if len(card_boxes) > 0 else h_collection
        
        print(f"[find_and_extract_first_row_cards] Row width: calculated={calculated_row_w}, limited to={row_w} (collection edge at {collection_right_edge})")
        
        # Extract the entire first row region
        first_row_img = collection_region_img[row_y:row_y+row_h, row_x:row_x+row_w].copy()
        
        # Apply 89% width fallback cropping if slider detection failed
        if use_fallback_crop:
            original_width = first_row_img.shape[1]
            cropped_width = int(original_width * 0.89)
            first_row_img = first_row_img[:, :cropped_width]
            print(f"[find_and_extract_first_row_cards] Applied 89% width fallback crop: {original_width} -> {cropped_width} pixels")
            
            # Update row_w for the visualization boxes
            row_w = cropped_width
        
        # Create a copy with bounding boxes drawn for visualization
        first_row_with_boxes = first_row_img.copy()
        for idx, (bx, by, bw, bh, _) in enumerate(card_boxes, 1):
            # Convert card box coordinates from collection_region_img to first_row_img coordinates
            # row_x = min_x - row_padding, so the offset is row_padding
            # The card box (bx, by) in collection coordinates becomes (bx - row_x, by - row_y) in row coordinates
            box_x_in_row = bx - row_x
            box_y_in_row = by - row_y
            
            # Ensure coordinates are within the row image bounds
            box_x_in_row = max(0, min(box_x_in_row, row_w - 1))
            box_y_in_row = max(0, min(box_y_in_row, row_h - 1))
            box_w_in_row = min(bw, row_w - box_x_in_row)
            box_h_in_row = min(bh, row_h - box_y_in_row)
            
            # Draw rectangle around detected card
            cv2.rectangle(first_row_with_boxes, 
                         (box_x_in_row, box_y_in_row), 
                         (box_x_in_row + box_w_in_row, box_y_in_row + box_h_in_row), 
                         (0, 255, 0), 2)  # Green rectangle
            
            # Add card number label
            cv2.putText(first_row_with_boxes, f"Card {idx}", 
                       (box_x_in_row + 5, box_y_in_row + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save the full row screenshot with bounding boxes
        row_output_path = output_dir / "first_row_full.png"
        cv2.imwrite(str(row_output_path), first_row_with_boxes)
        print(f"[find_and_extract_first_row_cards] Saved full first row screenshot with bounding boxes to {row_output_path}")
        print(f"[find_and_extract_first_row_cards] First row region: x={row_x}, y={row_y}, w={row_w}, h={row_h} (collection-relative)")
        
        # Also calculate and print screen coordinates for the full row
        screen_row_x = left + collection_start_x + row_x
        screen_row_y = top + collection_start_y + row_y
        screen_row_bottom_right_x = screen_row_x + row_w
        screen_row_bottom_right_y = screen_row_y + row_h
        print(f"[find_and_extract_first_row_cards] First row screen coordinates: ({screen_row_x}, {screen_row_y}) to ({screen_row_bottom_right_x}, {screen_row_bottom_right_y})")
    
    # Step 8: Extract and save each card with precise border detection
    print(f"[find_and_extract_first_row_cards] Saving individual cards to {output_dir}")
    
    for idx, (x, y, w, h, _) in enumerate(card_boxes, 1):
        # Extract a region slightly larger than the detected card to ensure we have context for border detection
        # Add padding to ensure we capture the full card even if detection was slightly off
        padding = int(collection_region_img.shape[1] * 0.02)  # 2% of collection width for padding
        region_x = max(0, x - padding)
        region_y = max(0, y - padding)
        region_w = min(w_collection - region_x, w + 2 * padding)
        region_h = min(h_collection - region_y, h + 2 * padding)
        
        # Extract the region containing the card
        card_region = collection_region_img[region_y:region_y+region_h, region_x:region_x+region_w].copy()
        
        # Detect the precise borders of the card within this region
        border_box = detect_card_borders(card_region)
        
        if border_box is None:
            # Fallback: if border detection fails, try a simple approach
            # Use the center portion of the detected region, slightly shrunk to avoid background
            print(f"[find_and_extract_first_row_cards] Warning: Could not detect borders for card {idx}, using conservative fallback")
            
            # Shrink the bounding box by 5% on each side to avoid background
            shrink_factor = 0.05
            shrink_x = int(w * shrink_factor)
            shrink_y = int(h * shrink_factor)
            
            border_x = padding + shrink_x
            border_y = padding + shrink_y
            border_w = max(int(w * 0.125), w - 2 * shrink_x)  # Minimum 12.5% of detected width
            border_h = max(int(h * 0.125), h - 2 * shrink_y)  # Minimum 12.5% of detected height
            
            # Ensure we don't exceed region bounds
            border_x = max(0, min(border_x, region_w - 1))
            border_y = max(0, min(border_y, region_h - 1))
            border_w = min(border_w, region_w - border_x)
            border_h = min(border_h, region_h - border_y)
        else:
            border_x, border_y, border_w, border_h = border_box
            # Ensure border box is within the region
            border_x = max(0, min(border_x, region_w - 1))
            border_y = max(0, min(border_y, region_h - 1))
            border_w = min(border_w, region_w - border_x)
            border_h = min(border_h, region_h - border_y)
        
        # Extract only the card area (within its borders)
        card_img = card_region[border_y:border_y+border_h, border_x:border_x+border_w].copy()
        
        # Resize card to proportional dimensions based on window size
        # Use smaller proportions to keep cards reasonably sized
        target_width = max(70, int(full_window_img.shape[1] * 0.045))  # ~4.5% of window width, min 70px
        target_height = max(100, int(full_window_img.shape[0] * 0.11))  # ~11% of window height, min 100px
        card_img = cv2.resize(card_img, (target_width, target_height), interpolation=cv2.INTER_AREA)
        
        # Calculate screen coordinates for the final card image (before resize)
        final_x = left + collection_start_x + region_x + border_x
        final_y = top + collection_start_y + region_y + border_y
        final_w = border_w
        final_h = border_h
        screen_bottom_right_x = final_x + final_w
        screen_bottom_right_y = final_y + final_h
        
        print(f"[find_and_extract_first_row_cards] Card {idx}:")
        print(f"  Detected size: {final_w}x{final_h} pixels")
        print(f"  Screen Top-Left: ({final_x}, {final_y})")
        print(f"  Screen Bottom-Right: ({screen_bottom_right_x}, {screen_bottom_right_y})")
        
        # Save card image (only the card, no background)
        output_path = output_dir / f"card_{idx:02d}.png"
        cv2.imwrite(str(output_path), card_img)
        print(f"[find_and_extract_first_row_cards] Saved card {idx} to {output_path}")
    
    print(f"[find_and_extract_first_row_cards] Successfully extracted and saved {len(card_boxes)} cards")
    return True

# ---------- Entrypoint ----------

def print_card_summary(cards_in_order: List[Tuple[str, int]]):
    """
    CHANGE 3: Print final summary of all cards and counts.
    Handles duplicate card name aggregation and maintains encounter order.
    """
    if not cards_in_order:
        print("\n=== FINAL CARD SUMMARY ===")
        print("No cards were successfully processed.")
        return
    
    print(f"\n=== FINAL CARD SUMMARY ===")
    print(f"Found {len(cards_in_order)} unique card(s) in encounter order:")
    print("-" * 60)
    
    total_cards = 0
    displayed_count = 0
    
    for i, (card_name, count) in enumerate(cards_in_order):
        # Debug: Show all entries being processed
        # print(f"[DEBUG] Entry {i+1}: '{card_name}' x{count}")
        
        # Only display non-empty card names
        if card_name and card_name.strip():
            displayed_count += 1
            print(f"{displayed_count}. {card_name}: x{count}")
            total_cards += count
        else:
            print(f"[DEBUG] Skipped empty card name at position {i+1}")
    
    print("-" * 60)
    # print(f"Total unique cards displayed: {displayed_count}")
    print(f"Total unique cards in list: {len(cards_in_order)}")
    print(f"Total card count: {total_cards}")
    
    # Additional debug info
    if displayed_count != len(cards_in_order):
        print(f"[DEBUG] Mismatch detected: {len(cards_in_order) - displayed_count} cards have empty names")

def main():
    """
    Modified main function to use the new card clicking and extraction functionality.
    CHANGE 2 & 3: Implements the complete workflow as specified.
    """
    print("Starting Master Duel Enhanced Collection Scanner...")
    print("This will detect the first 6 cards, click each one, and extract card information.")
    
    # Find game window
    win = find_game_window()
    if not win:
        print("Could not find Master Duel window. Please ensure the game is open and visible.")
        return
    
    print("Game window found. Starting card detection and clicking process...")
    
    # CHANGE 2: Use new multi-row function that clicks cards and extracts info
    card_summary = click_cards_and_extract_info_multi_row(win, max_rows=2)
    
    # CHANGE 3: Print final summary with aggregated counts
    print_card_summary(card_summary)
    
    print("\n=== Process Complete ===")
    print("The enhanced collection scanner has finished processing the first row of cards.")
    if card_summary:
        print("Check the output above for the complete card summary with counts.")
    else:
        print("No cards were successfully processed. Please check the game window and templates.")
    
    # Cleanup temporary files
    cleanup_temp_files()

def cleanup_temp_files():
    """Remove ALL temporary debugging files created during execution."""
    import glob
    
    # Find all tmp_rovodev_* files in current directory
    temp_files = glob.glob("tmp_rovodev_*")
    
    # Add specific known temporary files that might not match the pattern
    additional_temp_files = [
        "tmp_rovodev_full_window.png",
        "tmp_rovodev_card_area.png",
        "tmp_rovodev_name_region_debug.png", 
        "tmp_rovodev_name_gray_debug.png"
    ]
    
    # Combine and deduplicate
    all_temp_files = list(set(temp_files + additional_temp_files))
    
    removed_count = 0
    for file_path in all_temp_files:
        try:
            if Path(file_path).exists():
                Path(file_path).unlink()
                removed_count += 1
                print(f"[cleanup] Removed: {file_path}")
        except Exception as e:
            print(f"[cleanup] Could not remove {file_path}: {e}")
    
    if removed_count > 0:
        print(f"[cleanup] Removed {removed_count} temporary debugging files.")
    else:
        print("[cleanup] No temporary files found to remove.")

def main_original():
    print("Master Duel collection scraper - starting")
    
    # Find game window
    win = find_game_window(WINDOW_TITLE_KEYWORD)
    if not win:
        print("Could not find Master Duel window. Exiting.")
        return

    print(f"Detected window: {win.title} @ {win.left},{win.top} size {win.width}x{win.height}")
    
    # New functionality: Find and extract first row of 6 cards
    success = find_and_extract_first_row_cards(win)
    if success:
        print("[main] Successfully extracted first row of cards!")
    else:
        print("[main] Failed to extract first row of cards. Please check the error messages above.")
    
    # COMMENTED OUT: Old functionality for full collection scraping
    # # Load canonical DB
    # entries = load_ygojson_web_only()
    # matcher = CanonicalMatcher(entries)
    # 
    # print("Detecting collection grid region automatically...")
    # grid = detect_grid_for_window(win)
    # if not grid:
    #     print("Automatic detection failed. Please create a template image of the collection header and place it into ./templates/")
    #     # as fallback ask user to manually provide region
    #     try:
    #         print("Please move the mouse to the top-left of the grid and press Enter...")
    #         input()
    #         lx, ly = pyautogui.position()
    #         print("Now move the mouse to the bottom-right of the grid and press Enter...")
    #         input()
    #         rx, ry = pyautogui.position()
    #         grid = (min(lx, rx), min(ly, ry), abs(rx-lx), abs(ry-ly))
    #     except Exception:
    #         print("Manual region selection failed. Exiting.")
    #         return
    # 
    # print(f"Using grid region: {grid}")
    # print("Starting capture loop. Do not use the mouse or keyboard to interact with the game while capture runs.")
    # records = run_capture_loop(grid, matcher)
    # export_to_csv(records)

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
    Main capture+scroll loop (FAISS-based matching).
    - grid_region: (left, top, width, height) in screen coords.
    - matcher: kept for backward compatibility (object with .match(raw_name) -> (id,name,score))
               but by default uses matcher_loop_faiss() present in main.py (or imported).
    - Returns list of CardRecord objects.
    """
    import time
    import hashlib
    import pandas as pd
    import pyautogui
    from collections import deque

    seen_keys = set()
    records = []
    consecutive_no_new = 0
    prev_top_snapshot = None
    iterations = 0

    print(f"[run_capture_loop] Starting capture loop on region={grid_region}")

    start_time = time.time()

    # determine if FAISS matcher function is available in this module / namespace
    use_faiss_loop = False
    try:
        # matcher_loop_faiss should be defined in main.py (drop-in from prior step)
        _ = matcher_loop_faiss  # type: ignore[name-defined]
        use_faiss_loop = True
    except Exception:
        # if not present, we will fall back to the older per-thumb matcher.match(raw_name) method
        use_faiss_loop = False
        print("[run_capture_loop] Warning: matcher_loop_faiss not found — falling back to matcher.match per-thumb behavior.")

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

            # Split into thumbnail candidates (expected: list of (thumb_bgr, bbox))
            thumbs = split_grid_to_thumbs(crop)
            new_found = 0
            
            # Print thumbnail sizes and screen coordinates
            print(f"\n[Thumbnail Detection - Iteration {iterations}] Found {len(thumbs)} card thumbnails:")
            grid_left, grid_top, _, _ = grid_region
            for idx, (thumb_img, bbox) in enumerate(thumbs, 1):
                # bbox is (x, y, w, h) relative to the grid crop
                rel_x, rel_y, rel_w, rel_h = bbox
                
                # Convert to screen coordinates
                screen_top_left_x = grid_left + rel_x
                screen_top_left_y = grid_top + rel_y
                screen_bottom_right_x = grid_left + rel_x + rel_w
                screen_bottom_right_y = grid_top + rel_y + rel_h
                
                # Print size and coordinates
                print(f"  Card {idx}:")
                print(f"    Size: {rel_w}x{rel_h} pixels")
                print(f"    Screen Top-Left: ({screen_top_left_x}, {screen_top_left_y})")
                print(f"    Screen Bottom-Right: ({screen_bottom_right_x}, {screen_bottom_right_y})")
                print(f"    Relative to grid: ({rel_x}, {rel_y}) to ({rel_x + rel_w}, {rel_y + rel_h})")

            # If we have FAISS loop available, call it once with the full thumbnails list.
            results = []
            if use_faiss_loop:
                try:
                    # matcher_loop_faiss is expected to return a list of dicts aligned to thumbs order
                    results = matcher_loop_faiss(thumbnails=thumbs,
                                                 topk=6,
                                                 art_conf_thresh=0.75,
                                                 low_conf_thresh=0.55,
                                                 print_progress=False,
                                                 use_count_ocr=True)
                except Exception as e:
                    # If something goes wrong, fallback to per-thumb matching
                    print(f"[run_capture_loop] matcher_loop_faiss failed: {e}. Falling back to per-thumb matcher.")
                    use_faiss_loop = False
                    results = []

            # If the FAISS loop wasn't used (or failed), do the old per-thumb flow using matcher.match
            if not use_faiss_loop:
                for thumb_img, bbox in thumbs:
                    # fallback: run your OCR+matcher.match per-thumb (preserve original behavior)
                    try:
                        raw_name, count, _ = ocr_name_and_count(thumb_img)
                    except Exception:
                        raw_name, count = "", 1
                    try:
                        cid, cname, score = matcher.match(raw_name)
                    except Exception:
                        cid, cname, score = None, None, 0.0
                    out = {
                        "bbox": bbox,
                        "method": "legacy",
                        "card_id": cid,
                        "filename": cname,
                        "score": float(score or 0.0),
                        "count": int(count or 1),
                        "neighbors": [],
                        "raw_ocr": raw_name
                    }
                    results.append(out)

            # Process results into CardRecord entries, dedupe via seen_keys
            for entry in results:
                bbox = entry.get("bbox", (0,0,0,0))
                cid = entry.get("card_id")
                fname = entry.get("filename") or entry.get("raw_ocr") or ""
                score = float(entry.get("score") or 0.0)
                count = int(entry.get("count") or 1)

                # key: prefer canonical id, then filename, then raw_ocr; include count to avoid merging separate copies
                key_id = cid or fname or entry.get("raw_ocr") or ""
                key = (str(key_id), int(count))

                if key not in seen_keys:
                    seen_keys.add(key)
                    rec = CardRecord(canonical_id=cid,
                                     canonical_name=fname if fname else None,
                                     raw_name=entry.get("raw_ocr") or "",
                                     count=count,
                                     confidence=score,
                                     bbox=bbox)
                    records.append(rec)
                    new_found += 1

            # Top-slice similarity to detect repeated content after scroll
            top_row_h = min(int(crop.shape[0] * 0.13), crop.shape[0] // 6)  # 13% of crop height or 1/6, whichever is smaller
            top_slice = crop[0:top_row_h, :, :].copy()

            if prev_top_snapshot is not None and images_similar(prev_top_snapshot, top_slice):
                # top repeated; if also no new items, treat as no-new
                if new_found == 0:
                    consecutive_no_new += 1
            else:
                # content changed
                if new_found == 0:
                    # no new but content changed — keep consecutive_no_new unchanged (soft reset)
                    pass
                else:
                    consecutive_no_new = 0

            prev_top_snapshot = top_slice

            print(f"[capture #{iterations}] thumbs={len(thumbs)} new={new_found} total={len(records)} consecutive_no_new={consecutive_no_new}")
            recent_new = records[-new_found:] if new_found > 0 else []

            # live summary (reuse your existing function)
            try:
                live_progress_summary(records, iterations, start_time, recent_new)
            except Exception as e:
                # non-fatal if summary fails
                print(f"[run_capture_loop] live_progress_summary failed: {e}")

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

# FAISS version
# Put this in main.py (or import it) and call matcher_loop_faiss(thumbnails, ...)
# Requirements: art_match_faiss.py (from previous step), numpy, opencv-python, pytesseract (optional for fallback)
# pip install numpy opencv-python pytesseract
# Ensure ids.npy exists next to your FAISS index (created when building the index)

import os
import numpy as np
import cv2
import pytesseract
import difflib
from tqdm import tqdm

# try import art-match module (the script from earlier)
try:
    from art_match_faiss import match_thumbnail_bgr
except Exception as e:
    raise ImportError("Cannot import art_match_faiss.match_thumbnail_bgr — make sure art_match_faiss.py is in PATH and built artifacts exist.") from e

IDS_PATH = "ids.npy"  # created by build_faiss_index.py
# Optionally load mapping of index -> filename
try:
    _IDS = np.load(IDS_PATH, allow_pickle=True)
    _IDS_LIST = [str(x) for x in _IDS.tolist()]
except Exception:
    _IDS_LIST = None

# ----- small helper: extract a likely digits badge area and OCR digits only -----
def extract_count_from_thumb(thumb_bgr):
    """
    Attempt to extract a copy-count number from a thumbnail.
    Heuristic: many Master Duel UIs put a small badge in top-right or bottom-right.
    This attempts both regions and uses pytesseract to find digits.
    Returns integer count (default 1) and a confidence flag.
    """
    h, w = thumb_bgr.shape[:2]
    # check a small square in the top-right and bottom-right (tunable)
    candidates = [
        thumb_bgr[int(0.03*h):int(0.25*h), int(0.65*w):int(0.98*w)],  # top-right region
        thumb_bgr[int(0.6*h):int(0.98*h), int(0.65*w):int(0.98*w)],  # bottom-right region
    ]
    for crop in candidates:
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # threshold to isolate digits/badge
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # scale up to help OCR
        th = cv2.resize(th, (0,0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        # configure tesseract to only look for digits
        try:
            txt = pytesseract.image_to_string(th, config="--psm 7 -c tessedit_char_whitelist=0123456789")
            txt = txt.strip()
            if txt:
                # filter non-digits, return first numeric group
                import re
                m = re.search(r"(\d+)", txt)
                if m:
                    return int(m.group(1)), True
        except Exception:
            pass
    return 1, False

# ----- small helper: OCR name fallback and fuzzy-match against canonical names -----
def ocr_name_and_fuzzy_match(thumb_bgr, candidates_list=None, nbest=5):
    """
    Run a lightweight OCR on the thumbnail to get a name-like string, then fuzzy-match
    against the provided candidates_list (list of canonical filenames or names).
    If candidates_list is None, this will return the raw OCR string.
    Returns (best_match_id, best_match_name, score, raw_ocr).
    Score is difflib SequenceMatcher ratio (0..1), or 0 if no match.
    """
    # very small preproc to increase OCR quality for text
    gray = cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2GRAY)
    # focus on top half where name text is likely displayed (adjust if your UI shows it differently)
    h = gray.shape[0]
    crop = gray[int(0.45*h):int(0.75*h), :]  # tune depending on where name appears
    try:
        txt = pytesseract.image_to_string(crop, config="--psm 6")
        txt = txt.strip()
    except Exception:
        txt = ""
    if not txt:
        return None, None, 0.0, ""
    # Normalize OCR string
    raw = " ".join(txt.split())
    if not candidates_list:
        return None, None, 0.0, raw

    # Build a candidate display-name list (strip leading id_ and extension if present)
    canon_names = []
    mapping = {}
    for item in candidates_list:
        # item might be "12345_Card Name.jpg" or just "Card Name.jpg"
        name = item
        if "_" in item:
            # keep part after first underscore as candidate
            name = item.split("_", 1)[1]
        name = name.rsplit(".", 1)[0]
        canon_names.append(name)
        mapping[name] = item

    # Use difflib.get_close_matches first for speed
    matches = difflib.get_close_matches(raw, canon_names, n=nbest, cutoff=0.4)
    if matches:
        best = matches[0]
        # compute ratio
        ratio = difflib.SequenceMatcher(None, raw, best).ratio()
        matched_item = mapping[best]
        # try to extract id from matched_item if present
        card_id = matched_item.split("_", 1)[0] if "_" in matched_item else None
        return card_id, best, float(ratio), raw

    # fallback: do a brute force best ratio check
    best_ratio = 0.0
    best_name = None
    best_item = None
    for name in canon_names:
        r = difflib.SequenceMatcher(None, raw, name).ratio()
        if r > best_ratio:
            best_ratio = r
            best_name = name
            best_item = mapping[name]
    if best_name:
        card_id = best_item.split("_", 1)[0] if "_" in best_item else None
        return card_id, best_name, float(best_ratio), raw
    return None, None, 0.0, raw

# ----- main drop-in loop replacement -----
def matcher_loop_faiss(thumbnails,
                       topk=6,
                       art_conf_thresh=0.75,
                       low_conf_thresh=0.55,
                       print_progress=True,
                       use_count_ocr=True,
                       ids_list=_IDS_LIST):
    """
    thumbnails: iterable/list of (thumb_bgr, bbox) where thumb_bgr is an OpenCV BGR numpy array.
    Returns: list of dicts (one per thumbnail) with keys:
      - 'bbox' : original bbox
      - 'method': 'faiss'|'faiss_low'|'ocr_fallback'
      - 'card_id' (str or None)
      - 'filename' (str or None)  -- canonical filename or matched name
      - 'score' (float)  -- similarity or fuzzy-match score
      - 'count' (int)
      - 'neighbors' : list of (card_id, filename, score) from FAISS (topk) -- may be empty
      - 'raw_ocr': OCR text when OCR fallback used (may be "")
    """
    results = []
    iterator = thumbnails
    if print_progress:
        iterator = tqdm(thumbnails, desc="Matching thumbnails")

    for thumb_bgr, bbox in iterator:
        entry = {
            "bbox": bbox,
            "method": None,
            "card_id": None,
            "filename": None,
            "score": 0.0,
            "count": 1,
            "neighbors": [],
            "raw_ocr": ""
        }

        # 1) run FAISS matcher (expects BGR input per art_match_faiss design)
        try:
            neighbors = match_thumbnail_bgr(thumb_bgr, topk=topk)
        except Exception as e:
            neighbors = []
        # neighbors: list of (card_id, fname, score)
        if neighbors:
            entry["neighbors"] = neighbors
            top_card_id, top_fname, top_score = neighbors[0]
            entry["score"] = float(top_score)
            # Accept strong art matches
            if top_score >= art_conf_thresh:
                entry["method"] = "faiss"
                entry["card_id"] = str(top_card_id)
                entry["filename"] = str(top_fname)
            elif top_score >= low_conf_thresh:
                # borderline: return neighbors for voting or manual review
                entry["method"] = "faiss_low"
                entry["card_id"] = str(top_card_id)
                entry["filename"] = str(top_fname)
            else:
                # fallthrough to OCR fallback (name-based)
                entry["method"] = "faiss_no_good"
        else:
            entry["method"] = "no_faiss"

        # 2) count extraction (independent) -- do for all matches to get counts
        if use_count_ocr:
            try:
                count, count_conf = extract_count_from_thumb(thumb_bgr)
            except Exception:
                count, count_conf = 1, False
            entry["count"] = int(count if count and count>0 else 1)

        # 3) OCR fallback (only when no good art match)
        if entry["method"] in ("faiss_no_good", "no_faiss"):
            # try OCR name, fuzzy match against ids_list if available
            if ids_list:
                card_id, name, score, raw_ocr = ocr_name_and_fuzzy_match(thumb_bgr, candidates_list=ids_list, nbest=6)
            else:
                card_id, name, score, raw_ocr = ocr_name_and_fuzzy_match(thumb_bgr, candidates_list=None)
            entry["raw_ocr"] = raw_ocr
            if card_id and score > 0.45:
                entry["method"] = "ocr_fallback"
                entry["card_id"] = str(card_id)
                entry["filename"] = name
                entry["score"] = float(score)
            else:
                entry["method"] = "no_confident_match"

        # 4) print a short live status line (for monitoring)
        if print_progress:
            if entry["method"] in ("faiss", "faiss_low"):
                print_str = f"[{entry['method']}] id={entry['card_id']} score={entry['score']:.3f} count={entry['count']}"
            elif entry["method"] == "ocr_fallback":
                print_str = f"[ocr_fallback] id={entry['card_id']} score={entry['score']:.3f} raw='{entry['raw_ocr']}' count={entry['count']}"
            else:
                print_str = f"[NO_MATCH] count={entry['count']} raw_ocr='{entry['raw_ocr']}'"
            # use flush to make realtime monitoring easier
            print(print_str, flush=True)

        results.append(entry)

    return results

if __name__ == "__main__":
    main()
