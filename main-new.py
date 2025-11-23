#!/usr/bin/env python3
"""
Master Duel Collection Scraper - Optimized Version

Detects card collection grid, captures thumbnails, extracts card info via OCR,
and exports results. Optimized for speed and reduced computational complexity.
"""

import time
import signal
import sys
import re
import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from urllib.parse import quote
import requests

import cv2
import numpy as np
import pytesseract
import mss
import pyautogui
import pygetwindow as gw

# Configuration
SCROLL_DELAY = 0.1 # 0.15
BETWEEN_ROW_DELAY = 0.2 # 0.8
CLICK_DESC_DELAY = 0.1 # 0.5
WINDOW_TITLE_KEYWORD = "masterduel"
OUTPUT_CSV = "collection_output"
DEBUG = False
SUMMARY = True
CSV = True

# Template cache to avoid repeated loading
_TEMPLATE_CACHE = {}

# Global state for interruption handling
interruptible_cards = []
previous_first_card_name = None
previous_card_info = {}

class EndOfCollection(Exception):
    """Raised when end-of-collection is detected"""
    def __init__(self, partial=None):
        super().__init__("EndOfCollection")
        self.partial = partial or {}

# Utility Functions
def normalize_title(s: str) -> str:
    """Normalize window title for matching"""
    return re.sub(r"[^0-9a-z]", "", (s or "").lower())

def find_game_window(keyword: str = WINDOW_TITLE_KEYWORD) -> Optional[gw.Win32Window]:
    """Find Master Duel window by title"""
    try:
        wins = gw.getWindowsWithTitle(normalize_title(keyword))
    except Exception:
        all_windows = gw.getAllWindows()
        wins = [w for w in all_windows if keyword.lower() in w.title.lower()]
    if not wins:
        print("Could not find game window. Please ensure it is open and not minimized.")
        return None
    win = wins[0]
    if win.isMinimized:
        try:
            win.restore()
            time.sleep(0.3)
        except Exception:
            pass
    try:
        win.activate()
        time.sleep(0.2)
    except Exception:
        pass
    return win

def grab_region(region: Tuple[int, int, int, int]) -> np.ndarray:
    """Capture screen region using mss, return as BGR numpy array"""
    left, top, width, height = region
    try:
        with mss.mss() as sct:
            monitor = {
                "left": int(left),
                "top": int(top),
                "width": int(width),
                "height": int(height),
            }
            sct_img = sct.grab(monitor)
            arr = np.array(sct_img)
            if arr.ndim == 3 and arr.shape[2] == 4:
                img_bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
            elif arr.ndim == 3 and arr.shape[2] == 3:
                img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = arr.copy()
            return img_bgr
    except Exception as e:
        if DEBUG:
            print(f"Screen capture failed for region {region}: {e}")
        raise

def load_template_cached(template_path: Path) -> Optional[np.ndarray]:
    """Load template image with caching to avoid repeated disk I/O"""
    path_str = str(template_path)
    if path_str in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[path_str]
    if not template_path.exists():
        return None
    template = cv2.imread(path_str)
    if template is not None:
        _TEMPLATE_CACHE[path_str] = template
    return template

# Card Detection Functions
def detect_card_borders(card_region_img: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Detect card borders using optimized edge detection. Returns (x, y, w, h) or None"""
    if card_region_img.size == 0:
        return None
    h, w = card_region_img.shape[:2]
    region_area = w * h
    if len(card_region_img.shape) == 3:
        gray = cv2.cvtColor(card_region_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = card_region_img.copy()
    
    # Optimized single-pass edge detection with morphological operations
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Find best rectangular contour
    candidates = []
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
                    candidates.append((x, y, cw, ch, extent))
    
    if candidates:
        candidates.sort(key=lambda c: c[4], reverse=True)
        x, y, cw, ch, _ = candidates[0]
        padding = 3
        x = max(0, x - padding)
        y = max(0, y - padding)
        cw = min(w - x, cw + 2 * padding)
        ch = min(h - y, ch + 2 * padding)
        return (x, y, cw, ch)
    return None

def ocr_description_zone_card_info(desc_zone_img: np.ndarray, lang: str = "eng", 
                                   card_number: int = 0, row_number: int = 1,
                                   return_count_header: bool = False) -> Tuple[str, int, int]:
    """Extract card name and count from description zone using OCR"""
    if desc_zone_img is None or desc_zone_img.size == 0:
        return ("", 1, 0) if return_count_header else ("", 1)
    
    h, w = desc_zone_img.shape[:2]
    name_h = int(h * 0.25)
    symbol_margin = int(w * 0.15)
    left_margin = int(w * 0.02)
    top_margin = int(name_h * 0.1)
    
    # Optimized name region extraction
    current_height = name_h - 2 - top_margin
    current_width = w - symbol_margin - left_margin
    height_reduction = int(current_height * 0.3)
    width_reduction = int(current_height * 0.017)
    refined_top = top_margin + height_reduction // 2
    refined_bottom = name_h - 2 - (height_reduction - height_reduction // 2)
    refined_left = left_margin + width_reduction // 2
    refined_right = w - symbol_margin - (width_reduction - width_reduction // 2)
    additional_right_reduction = int(current_width * 0.017)
    refined_right = refined_right - additional_right_reduction
    
    name_region = desc_zone_img[refined_top:refined_bottom, refined_left:refined_right]
    
    # OCR for card name
    card_name = ""
    if name_region.size > 0:
        name_gray = cv2.cvtColor(name_region, cv2.COLOR_BGR2GRAY)
        name_gray = cv2.resize(name_gray, (name_gray.shape[1] * 3, name_gray.shape[0] * 3), 
                               interpolation=cv2.INTER_CUBIC)
        _, name_thresh = cv2.threshold(name_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        try:
            card_name = pytesseract.image_to_string(name_thresh, lang=lang, 
                                                    config=r"--oem 3 --psm 7").strip()
            card_name = card_name.replace("\n", " ").replace("\x0c", "").strip()
        except Exception:
            card_name = ""
    
    # Count extraction using template matching
    count = 1
    header_x = 0
    count_header_path = Path("templates/count_header.PNG")
    count_header_template = load_template_cached(count_header_path)
    
    if count_header_template is not None:
        gray_desc_zone = cv2.cvtColor(desc_zone_img, cv2.COLOR_BGR2GRAY)
        gray_count_header = cv2.cvtColor(count_header_template, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray_desc_zone, gray_count_header, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
        if max_val >= 0.6:
            header_x, header_y = max_loc
            header_h, header_w = gray_count_header.shape[:2]
            count_region_y = header_y + header_h + 2
            initial_count_region_w = header_w + 20
            width_reduction = int(initial_count_region_w * 0.43)
            count_region_w = initial_count_region_w - width_reduction
            count_region_h = min(30, h - count_region_y)
            
            if count_region_y + count_region_h <= h and header_x + count_region_w <= w:
                count_region = desc_zone_img[count_region_y:count_region_y + count_region_h,
                                            header_x:header_x + count_region_w]
                if count_region.size > 0:
                    count_gray = cv2.cvtColor(count_region, cv2.COLOR_BGR2GRAY)
                    count_gray = cv2.resize(count_gray, (count_gray.shape[1] * 3, count_gray.shape[0] * 3),
                                           interpolation=cv2.INTER_CUBIC)
                    _, count_thresh1 = cv2.threshold(count_gray, 120, 255, cv2.THRESH_BINARY_INV)
                    _, count_thresh2 = cv2.threshold(count_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    cfg_count = r"--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789x"
                    
                    for thresh_img in [count_thresh1, count_thresh2]:
                        try:
                            txt = pytesseract.image_to_string(thresh_img, config=cfg_count).strip()
                            if "x" in txt.lower():
                                num_part = txt.lower().split("x")[-1]
                                if num_part.isdigit():
                                    count = int(num_part)
                                    break
                            elif txt.isdigit():
                                count = int(txt)
                                break
                        except Exception:
                            continue
    
    if return_count_header:
        return card_name, count, header_x
    return card_name, count

def analyze_dism_area(dism_area_img: np.ndarray) -> int:
    """Analyze dism area for dustable value using OCR"""
    if dism_area_img is None or dism_area_img.size == 0:
        return 0
    try:
        dism_gray = cv2.cvtColor(dism_area_img, cv2.COLOR_BGR2GRAY)
        dism_gray = cv2.resize(dism_gray, (dism_gray.shape[1] * 3, dism_area_img.shape[0] * 3),
                               interpolation=cv2.INTER_CUBIC)
        _, dism_thresh = cv2.threshold(dism_gray, 120, 255, cv2.THRESH_BINARY_INV)
        cfg_dism = r"--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789"
        txt = pytesseract.image_to_string(dism_thresh, config=cfg_dism).strip()
        if txt.isdigit():
            return int(txt)
    except Exception:
        pass
    return 0

def get_card_info(canonical_name: str) -> dict:
    """Fetch full card info from YGOPRODECK API using canonical name"""
    if not canonical_name:
        return {}
    url = f"https://db.ygoprodeck.com/api/v7/cardinfo.php?name={quote(canonical_name)}&misc=yes"
    try:
        r = requests.get(url, timeout=5.0)
        r.raise_for_status()
        data = r.json()
        if "data" in data and data["data"]:
            return data["data"][0]
    except Exception:
        pass
    return {}

def get_card_rarity_from_info(card_info: dict) -> str:
    """Extract rarity string from card info dict"""
    misc_info = card_info.get("misc_info", [])
    for misc in misc_info:
        if "md_rarity" in misc:
            md_rarity = misc["md_rarity"]
            if md_rarity == "Common":
                return "N "
            elif md_rarity == "Rare":
                return "R "
            elif md_rarity == "Super Rare":
                return "SR"
            elif md_rarity == "Ultra Rare":
                return "UR"
    return "N "

def prepare_csv_data(cards_in_order) -> List:
    """Prepare CSV data from cards_in_order"""
    print("Preparing CSV file data...")
    csv_data = []
    try:
        from card_name_matcher import get_canonical_name_and_legacy_status
        use_matcher = True
    except Exception:
        use_matcher = False
    for idx, (card_name, count_list, dustable_value) in enumerate(cards_in_order):
        if use_matcher:
            canonical_name, is_legacy_pack = get_canonical_name_and_legacy_status(card_name)
        else:
            canonical_name, is_legacy_pack = card_name, False
        if not canonical_name or canonical_name.strip() == "":
            canonical_name = card_name
        display_name = canonical_name
        legacy_status = bool(is_legacy_pack)
        card_info = get_card_info(canonical_name)
        card_rarity = get_card_rarity_from_info(card_info)
        # Update rarity in count_list
        for item in count_list:
            item[3] = card_rarity
        # Compute category counts
        basic_count = 0
        glossy_count = 0
        royal_count = 0
        for count, x_coord, desc_zone_width, _ in count_list:
            category = determine_card_category(x_coord, desc_zone_width)
            if category == "Basic":
                basic_count += count
            elif category == "Glossy":
                glossy_count += count
            elif category == "Royal":
                royal_count += count
        total_copies = sum(count for count, _, _, _ in count_list)
        copies_str = f"{total_copies} [Basic {basic_count}, Glossy {glossy_count}, Royal {royal_count}]"
        # Rarity token
        first_rarity = card_rarity.strip()
        if first_rarity in ("N", "R"):
            visible_token = first_rarity + " "
        else:
            visible_token = first_rarity
        # Determine Card Type based on frameType
        frame_type = card_info.get("frameType", "")
        if frame_type == "trap":
            card_type = "Trap"
        elif frame_type == "spell":
            card_type = "Spell"
        else:
            card_type = "Monster"

        # Card Stats
        card_stats = ""
        if card_info.get("type") not in ["Trap Card", "Spell Card"]:
            if frame_type == "xyz":
                level = card_info.get("level", "")
                card_stats = f" Rank {level}, "
            elif frame_type == "link":
                linkval = card_info.get("linkval", "")
                card_stats = f" Link {linkval}, "
            else:
                level = card_info.get("level", "")
                card_stats = f" Level {level}, "
            attribute = card_info.get("attribute", "")
            race = card_info.get("race", "")
            atk = card_info.get("atk", "")
            def_val = card_info.get("def")
            def_str = "-" if def_val is None else str(def_val)
            card_stats += f"{attribute}, {race}, {atk}/{def_str}"

        csv_row = {
            "rarity": visible_token,
            "name": display_name,
            "legacy": "Yes" if legacy_status else "No",
            "copies": copies_str,
            "dustable": dustable_value,
            "archetype": card_info.get("archetype") or None,
            "card_type": card_type,
            "subtype": card_info.get("humanReadableCardType", ""),
            "card_stats": card_stats,
            "effect": card_info.get("desc", ""),
        }
        csv_data.append(csv_row)
    return csv_data

def write_csv(csv_data, message=None):
    """Write CSV data to file"""
    print("Saving collection to CSV file...")
    try:
        import csv
        csv_dir = Path("collection_csv")
        csv_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_[%H%M]")
        base_name = OUTPUT_CSV
        filename = f"{base_name}_{timestamp}.csv"
        filepath = csv_dir / filename
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["Rarity", "Name", "Legacy Pack", "Copies", "Dustable", "Archetype", "Card Frame", "Card Type", "Card Stats", "Effect"])
            writer.writeheader()
            for row in csv_data:
                writer.writerow({
                    "Rarity": row["rarity"],
                    "Name": row["name"],
                    "Legacy Pack": row["legacy"],
                    "Copies": row["copies"],
                    "Dustable": row["dustable"],
                    "Archetype": row["archetype"],
                    "Card Frame": row["card_type"],
                    "Card Type": row["subtype"],
                    "Card Stats": row["card_stats"],
                    "Effect": row["effect"],
                })
        final_message = message or f"Results exported to {filepath}"
        if "Partial" in final_message:
            final_message = final_message.replace(OUTPUT_CSV, str(filepath))
        print(final_message)
    except Exception as e:
        print(f"Failed to export CSV: {e}")

def determine_card_category(x_coord: int, desc_zone_width: float) -> str:
    """Determine card category (Basic/Glossy/Royal) based on count header position"""
    basic_threshold = 0.735 * desc_zone_width
    glossy_threshold = 0.811 * desc_zone_width
    royal_threshold = 0.884 * desc_zone_width
    distances = [
        (abs(x_coord - basic_threshold), "Basic"),
        (abs(x_coord - glossy_threshold), "Glossy"),
        (abs(x_coord - royal_threshold), "Royal"),
    ]
    _, closest_category = min(distances, key=lambda item: item[0])
    return closest_category

# Main Processing Functions
def detect_full_collection_area(win):
    """Detect and return full collection area coordinates"""
    left, top, width, height = win.left, win.top, win.width, win.height
    if left < 0 or top < 0:
        try:
            win.moveTo(8, 8)
            time.sleep(0.2)
            left, top, width, height = win.left, win.top, win.width, win.height
        except Exception:
            pass
    
    try:
        full_window_img = grab_region((left, top, width, height))
    except Exception as e:
        if DEBUG:
            print(f"Failed to grab window region: {e}")
        return None, None
    
    header_template = load_template_cached(Path("templates/header.PNG"))
    if header_template is None:
        return None, None
    
    gray_window = cv2.cvtColor(full_window_img, cv2.COLOR_BGR2GRAY)
    gray_header = cv2.cvtColor(header_template, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray_window, gray_header, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    
    if max_val < 0.5:
        return None, None
    
    header_x, header_y = max_loc
    header_h, header_w = gray_header.shape[:2]
    collection_area_margin = int(header_w * 0.02)
    collection_area_x = header_x + collection_area_margin
    collection_area_w = int(header_w * 0.935)
    collection_area_y = header_y + header_h + 10
    collection_area_h = height - collection_area_y - 50
    card_width = collection_area_w // 6
    card_height = collection_area_h // 5
    
    return (collection_area_x, collection_area_y, collection_area_w, collection_area_h), (card_width, card_height)

def click_cards_and_extract_info_single_row(win, row_number: int = 1,
                                            collection_coords=None,
                                            card_dims=None) -> Dict[str, int]:
    """Process a single row of 6 cards, extracting card info"""
    global previous_first_card_name, previous_card_info
    
    left, top, width, height = win.left, win.top, win.width, win.height
    if left < 0 or top < 0:
        try:
            win.moveTo(8, 8)
            time.sleep(0.2)
            left, top, width, height = win.left, win.top, win.width, win.height
        except Exception:
            pass
    
    try:
        full_window_img = grab_region((left, top, width, height))
    except Exception:
        return {}
    
    header_template = load_template_cached(Path("templates/header.PNG"))
    if header_template is None:
        return {}
    
    gray_window = cv2.cvtColor(full_window_img, cv2.COLOR_BGR2GRAY)
    gray_header = cv2.cvtColor(header_template, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray_window, gray_header, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    
    if max_val < 0.5:
        return {}
    
    header_x, header_y = max_loc
    header_h, header_w = gray_header.shape[:2]
    
    if collection_coords is not None:
        card_area_x, card_area_y, card_area_w, card_area_h = map(int, collection_coords)
    else:
        card_area_margin = int(header_w * 0.02)
        card_area_x = header_x + card_area_margin
        card_area_w = int(header_w * 0.935)
        card_area_y = header_y + header_h + 10
        card_area_h = height - card_area_y - 50
    
    estimated_card_width = card_area_w // 6
    estimated_card_height = int(estimated_card_width * 1.4)
    card_area_h = min(estimated_card_height + 20, height - card_area_y)
    
    if card_area_w <= 0 or card_area_h <= 0:
        return {}
    
    card_area_img = full_window_img[card_area_y:card_area_y + card_area_h, 
                                    card_area_x:card_area_x + card_area_w]
    area_h, area_w = card_area_img.shape[:2]
    card_width = area_w // 6
    card_height = min(area_h, int(card_width * 1.4))
    start_y = max(0, (area_h - card_height) // 2)
    
    cards_to_process = []
    for i in range(6):
        card_x = i * card_width
        card_y = start_y
        margin_x = int(card_width * 0.05)
        margin_y = int(card_height * 0.05)
        final_x = max(0, card_x + margin_x)
        final_y = max(0, card_y + margin_y)
        final_w = min(area_w - final_x, card_width - 2 * margin_x)
        final_h = min(area_h - final_y, card_height - 2 * margin_y)
        if final_w > 10 and final_h > 10:
            card_img = card_area_img[final_y:final_y + final_h, final_x:final_x + final_w].copy()
            cards_to_process.append((card_img, (final_x, final_y, final_w, final_h)))
    
    card_summary = {}
    
    def detect_and_capture_description_zone(window_img: np.ndarray) -> Optional[np.ndarray]:
        """Detect card description zone using template matching"""
        card_header_template = load_template_cached(Path("templates/card_header.PNG"))
        if card_header_template is None:
            return None
        gray_window = cv2.cvtColor(window_img, cv2.COLOR_BGR2GRAY)
        gray_card_header = cv2.cvtColor(card_header_template, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray_window, gray_card_header, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val < 0.4:
            return None
        header_x, header_y = max_loc
        header_h, header_w = gray_card_header.shape[:2]
        desc_zone_x = header_x
        desc_zone_y = header_y + header_h + 5
        window_h, window_w = window_img.shape[:2]
        desc_zone_w = min(int(window_w * 0.22), header_w * 3)
        original_desc_zone_h = int(window_h * 0.16)
        desc_zone_h = int(original_desc_zone_h * 1.73)
        desc_zone_w = min(desc_zone_w, window_w - desc_zone_x)
        desc_zone_h = min(desc_zone_h, window_h - desc_zone_y)
        if desc_zone_w <= 0 or desc_zone_h <= 0:
            return None
        return window_img[desc_zone_y:desc_zone_y + desc_zone_h, 
                         desc_zone_x:desc_zone_x + desc_zone_w].copy()
    
    for i, (card_img, card_bbox) in enumerate(cards_to_process):
        card_x, card_y, card_w, card_h = card_bbox
        click_x = left + card_area_x + card_x + card_w // 2
        click_y = top + card_area_y + card_y + card_h // 2
        
        try:
            pyautogui.click(click_x, click_y)
            time.sleep(CLICK_DESC_DELAY)  # Reduced from 0.8s
            clicked_window_img = grab_region((left, top, width, height))
            desc_zone_img = detect_and_capture_description_zone(clicked_window_img)
            
            if desc_zone_img is not None:
                h_desc, w_desc = desc_zone_img.shape[:2]
                original_dism_width = int(w_desc * 0.15)
                original_dism_height = int(h_desc * 0.15)
                original_dism_x = w_desc - original_dism_width
                original_dism_y = h_desc - original_dism_height
                reduction_width = int(original_dism_height * 0.33)
                reduction_height = int(original_dism_height * 0.125)
                refined_width = original_dism_width - reduction_width
                refined_height = original_dism_height - reduction_height
                dism_area_img = desc_zone_img[original_dism_y:original_dism_y + refined_height,
                                             original_dism_x:original_dism_x + refined_width].copy()
                
                card_name, count, count_header_x = ocr_description_zone_card_info(
                    desc_zone_img, card_number=i + 1, row_number=row_number,
                    return_count_header=True
                )
                card_rarity = ""
                dustable_value = analyze_dism_area(dism_area_img)
                
                # Phase 2 early termination logic
                if row_number > 4:
                    if i == 0:
                        if previous_first_card_name is not None and card_name == previous_first_card_name:
                            previous_card_info = {}
                            raise EndOfCollection(partial=card_summary)
                        current_row_first_card = card_name
                    else:
                        if (i - 1) in previous_card_info:
                            prev_name, prev_header_x = previous_card_info[i - 1]
                            if card_name == prev_name and count_header_x == prev_header_x:
                                previous_card_info = {}
                                raise EndOfCollection(partial=card_summary)
                    if i < 5:
                        previous_card_info[i] = (card_name, count_header_x)
                    elif i == 5:
                        previous_card_info = {}
                
                if card_name:
                    desc_zone_width = w_desc
                    if card_name in card_summary:
                        card_summary[card_name][0].append(
                            [count, count_header_x, desc_zone_width, card_rarity]
                        )
                        card_summary[card_name] = (
                            card_summary[card_name][0],
                            max(card_summary[card_name][1], dustable_value),
                        )
                    else:
                        card_summary[card_name] = (
                            [[count, count_header_x, desc_zone_width, card_rarity]],
                            dustable_value,
                        )
        except EndOfCollection:
            raise
        except Exception:
            continue
        
        if row_number > 4 and i < 6:
            previous_card_info[i] = (card_name, count_header_x)
        elif i == 6:
            previous_card_info.clear()
    
    if row_number > 4 and "current_row_first_card" in locals() and current_row_first_card:
        previous_first_card_name = current_row_first_card
    return card_summary

def process_full_collection_phases(win) -> List:
    """Process collection in two phases: Phase 1 (rows 1-4), Phase 2 (rows 5+)"""
    collection_coords, card_dims = detect_full_collection_area(win)
    if collection_coords is None or card_dims is None:
        return []
    
    collection_area_x, collection_area_y, collection_area_w, collection_area_h = collection_coords
    scroll_pattern = [1, 2, 1, 2, 1, 1, 2]
    cards_in_order = []
    
    # Phase 1: Process first 4 rows
    for row_num in range(1, 5):
        row_offset = int(collection_area_h * 0.20 * (row_num - 1))
        row_collection_coords = (
            collection_area_x,
            collection_area_y + row_offset,
            collection_area_w,
            collection_area_h - row_offset,
        )
        row_summary = click_cards_and_extract_info_single_row(
            win, row_number=row_num, collection_coords=row_collection_coords, 
            card_dims=card_dims
        )
        
        for card_name, (count_list, dustable_value) in row_summary.items():
            if not card_name or card_name.strip() == "":
                continue
            found_existing = False
            for i, (existing_name, existing_count_list, existing_dustable) in enumerate(cards_in_order):
                if existing_name == card_name:
                    new_dustable = max(existing_dustable, dustable_value)
                    combined_count_list = existing_count_list + count_list
                    cards_in_order[i] = (existing_name, combined_count_list, new_dustable)
                    found_existing = True
                    break
            if not found_existing:
                cards_in_order.append((card_name, count_list, dustable_value))
        update_interruptible_cards(cards_in_order)
    
    # Phase 2: Process rows 5+ with scrolling
    phase2_start_offset = int(collection_area_h * 0.20 * 4)
    phase2_collection_coords = (
        collection_area_x,
        collection_area_y + phase2_start_offset,
        collection_area_w,
        max(50, collection_area_h - phase2_start_offset),
    )
    
    phase2_start_row = 5
    while True:
        try:
            for idx in range(8):
                row_num = phase2_start_row + idx
                if idx > 0:
                    if idx - 1 < len(scroll_pattern):
                        scroll_count = scroll_pattern[idx - 1]
                        for _ in range(scroll_count):
                            pyautogui.scroll(-1)
                            time.sleep(SCROLL_DELAY)  # Reduced from 0.3s
                    time.sleep(BETWEEN_ROW_DELAY)  # Reduced from 1.5s
                
                if idx == 7:
                    continue
                
                try:
                    row_summary = click_cards_and_extract_info_single_row(
                        win, row_number=row_num, collection_coords=phase2_collection_coords,
                        card_dims=card_dims
                    )
                except EndOfCollection as e:
                    if getattr(e, "partial", None):
                        for cname, (count_list, dust_value) in e.partial.items():
                            found_existing = False
                            for j, (existing_name, existing_count_list, existing_dustable) in enumerate(cards_in_order):
                                if existing_name == cname:
                                    new_dustable = max(existing_dustable, dust_value)
                                    combined_count_list = existing_count_list + count_list
                                    cards_in_order[j] = (existing_name, combined_count_list, new_dustable)
                                    found_existing = True
                                    break
                            if not found_existing:
                                cards_in_order.append((cname, count_list, dust_value))
                    update_interruptible_cards(cards_in_order)
                    return cards_in_order
                
                for card_name, (count_list, dustable_value) in row_summary.items():
                    if not card_name or card_name.strip() == "":
                        continue
                    found_existing = False
                    for k, (existing_name, existing_count_list, existing_dustable) in enumerate(cards_in_order):
                        if existing_name == card_name:
                            new_dustable = max(existing_dustable, dustable_value)
                            combined_count_list = existing_count_list + count_list
                            cards_in_order[k] = (existing_name, combined_count_list, new_dustable)
                            found_existing = True
                            break
                    if not found_existing:
                        cards_in_order.append((card_name, count_list, dustable_value))
                update_interruptible_cards(cards_in_order)
            phase2_start_row += 8
        except Exception:
            break
    return cards_in_order

# Output Functions
def print_card_summary(cards_in_order: List):
    """Print final card summary with color-coded output"""
    print("Finished processing cards. Preparing final summary...")
    _ansi_re = re.compile(r"\x1b\[[0-9;]*m")
    def strip_ansi(s: str) -> str:
        return _ansi_re.sub("", s) if isinstance(s, str) else ""
    def visible_len(s: str) -> int:
        return len(strip_ansi(s))
    def pad_right_ansi(s: str, width: int) -> str:
        v = visible_len(s)
        return s if v >= width else s + (" " * (width - v))
    
    # ANSI colors
    LIGHT_BLUE = "\033[94m"
    BASIC_COLOR = "\033[97m"
    GLOSSY_COLOR = "\033[92m"
    ROYAL_COLOR = "\033[93m"
    LIGHT_RED = "\033[91m"
    RESET = "\033[0m"
    UNDERLINE_WHITE = "\033[4;97m"
    RARITY_WHITE = "\033[1;97m"
    RARITY_LIGHT_BLUE = "\033[1;96m"
    RARITY_GOLD = "\033[1;93m"
    RARITY_DARK_BLUE = "\033[1;94m"
    RARITY_BOLD = "\033[1m"
    
    def rarity_token_to_color(token: str) -> str:
        t = (token or "").strip()
        if t == "N":
            return RARITY_WHITE
        if t == "R":
            return RARITY_LIGHT_BLUE
        if t == "SR":
            return RARITY_GOLD
        if t == "UR":
            return RARITY_DARK_BLUE
        return RARITY_BOLD
    
    cards_in_order = [(name, cl, dv) for name, cl, dv in cards_in_order if name and name.strip()]
    if not cards_in_order:
        print("\n=== FINAL CARD SUMMARY ===\nNo cards were successfully processed.")
        return
    
    try:
        from card_name_matcher import get_canonical_name_and_legacy_status
        use_canonical_names = True
    except Exception:
        use_canonical_names = False
    
    gem_pack_cards = []
    legacy_pack_cards = []

    for idx, (card_name, count_list, dustable_value) in enumerate(cards_in_order):
        if use_canonical_names:
            try:
                canonical_name, is_legacy_pack = get_canonical_name_and_legacy_status(card_name)
            except Exception:
                canonical_name, is_legacy_pack = card_name, False
            if not canonical_name or canonical_name.strip() == "":
                canonical_name = card_name
            display_name = canonical_name
            legacy_status = bool(is_legacy_pack)
        else:
            display_name = card_name
            legacy_status = False

        # Get full card info and rarity
        card_info = get_card_info(canonical_name)
        card_rarity = get_card_rarity_from_info(card_info)
        # Update rarity in count_list
        for item in count_list:
            item[3] = card_rarity

        # Compute category counts for CSV
        basic_count = 0
        glossy_count = 0
        royal_count = 0
        for count, x_coord, desc_zone_width, _ in count_list:
            category = determine_card_category(x_coord, desc_zone_width)
            if category == "Basic":
                basic_count += count
            elif category == "Glossy":
                glossy_count += count
            elif category == "Royal":
                royal_count += count
        total_copies = sum(count for count, _, _, _ in count_list)
        copies_str = f"{total_copies} [Basic {basic_count}, Glossy {glossy_count}, Royal {royal_count}]"

        counts_str_elems = []
        for count, x_coord, desc_zone_width, rarity in count_list:
            category = determine_card_category(x_coord, desc_zone_width)
            if category == "Basic":
                colored_category = f"{BASIC_COLOR}{category}{RESET}"
            elif category == "Glossy":
                colored_category = f"{GLOSSY_COLOR}{category}{RESET}"
            elif category == "Royal":
                colored_category = f"{ROYAL_COLOR}{category}{RESET}"
            else:
                colored_category = category
            counts_str_elems.append(f"{colored_category} x{count}")
        counts_str = ", ".join(counts_str_elems)

        first_rarity = card_rarity
        
        rarity_color = rarity_token_to_color(first_rarity)
        visible_token = first_rarity
        colored_rarity_with_brackets = f"{rarity_color}{RARITY_BOLD}[{visible_token}]{RESET}"
        colored_name = f"{rarity_color}{RARITY_BOLD}{display_name}{RESET}"
        card_total = sum(int(count) for count, _, _, _ in count_list)
        
        card_info = {
            "orig_index": idx,
            "display_name": display_name,
            "colored_name": colored_name,
            "plain_name": display_name,
            "colored_rarity_with_brackets": colored_rarity_with_brackets,
            "counts_str": counts_str,
            "dustable_value": dustable_value,
            "card_total": card_total,
            "first_rarity_token": visible_token,
        }



        if legacy_status:
            legacy_pack_cards.append((idx, card_info))
        else:
            gem_pack_cards.append((idx, card_info))
    
    combined_total = max(1, len(gem_pack_cards) + len(legacy_pack_cards))
    idx_padding_width = len(str(combined_total))
    all_rows = gem_pack_cards + legacy_pack_cards
    min_rarity_w = 3
    min_name_w = 10
    min_counts_w = 6
    min_dust_w = 1
    max_rarity_w = min_rarity_w
    max_name_w = min_name_w
    max_counts_w = min_counts_w
    max_dust_w = min_dust_w
    
    for _, info in all_rows:
        max_rarity_w = max(max_rarity_w, visible_len(info["colored_rarity_with_brackets"]))
        max_name_w = max(max_name_w, visible_len(info["colored_name"]))
        max_counts_w = max(max_counts_w, visible_len(info["counts_str"]))
        max_dust_w = max(max_dust_w, len(str(info["dustable_value"])))
    
    print(f"\n=== FINAL CARD SUMMARY ===")
    print(f"Found {len(cards_in_order)} unique card(s) in encounter order:")
    
    def _print_section(title: str, rows):
        if not rows:
            return
        print(f"\n{UNDERLINE_WHITE}{title}{RESET}")
        for local_idx, (_, info) in enumerate(rows, start=1):
            idx_str = str(local_idx).zfill(idx_padding_width)
            idx_display = f"{idx_str}."
            rarity_field = pad_right_ansi(info["colored_rarity_with_brackets"], max_rarity_w)
            name_field = pad_right_ansi(info["colored_name"], max_name_w)
            counts_field = pad_right_ansi(info["counts_str"], max_counts_w)
            dust_field = str(info["dustable_value"]).rjust(max_dust_w)
            line = (
                f"{idx_display} {rarity_field} {name_field} | "
                f"{LIGHT_BLUE}COPIES{RESET}: {counts_field} | "
                f"{LIGHT_RED}DUSTABLE{RESET}: x{dust_field}"
            )
            print(line)
    
    _print_section("Gem Pack & Structure Deck", gem_pack_cards)
    _print_section("Legacy Pack", legacy_pack_cards)
    
    total_cards = sum(info["card_total"] for _, info in gem_pack_cards + legacy_pack_cards)
    total_unique = len(gem_pack_cards) + len(legacy_pack_cards)
    print(f"Total unique cards: {total_unique}")
    print(f"Total card count: {total_cards}")
    if gem_pack_cards:
        print(f"Gem Pack & Structure Deck: {len(gem_pack_cards)} unique cards")
    if legacy_pack_cards:
        print(f"Legacy Pack: {len(legacy_pack_cards)} unique cards")

    # Export to CSV if enabled
    if CSV:
        csv_data = prepare_csv_data(cards_in_order)
        if csv_data:
            write_csv(csv_data)

def signal_handler(sig, frame, cards_in_order_ref=None):
    """Handle Ctrl+C interruption gracefully"""
    print("\n\n=== SCRIPT INTERRUPTED ===")
    print("Processing was cancelled. Here is the summary of cards analyzed so far:")
    print("-" * 50)
    cards = cards_in_order_ref[0] if cards_in_order_ref and cards_in_order_ref[0] else interruptible_cards
    if SUMMARY:
        if cards:
            print_card_summary(cards)
        else:
            print("No cards were analyzed before interruption.")
    else:
        print("Summary printing disabled.")
    # Export partial CSV if enabled
    if CSV and cards:
        csv_data = prepare_csv_data(cards)
        if csv_data:
            write_csv(csv_data, f"Partial results exported to {OUTPUT_CSV}")
    print("\n=== PROCESSING STOPPED ===")
    sys.exit(0)

def update_interruptible_cards(cards_in_order):
    """Update global interruptible cards list for signal handler"""
    global interruptible_cards
    interruptible_cards = cards_in_order[:]

def main():
    """Main entry point - process Master Duel collection"""
    print("Starting Master Duel Collection Exporter...")
    print("- Phase 1: Process first 4 rows without scrolling")
    print("- Phase 2: Process remaining rows with scrolling")
    print("Press Ctrl+C at any time to stop and see current results.")
    
    cards_container = [[]]
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, cards_container))
    
    win = find_game_window()
    if not win:
        print("Could not find Master Duel window. Please ensure the game is open and visible.")
        return
    
    print("Game window found. Starting card detection and clicking process...")
    
    try:
        cards_in_order = process_full_collection_phases(win)
        cards_container[0] = cards_in_order
        if SUMMARY:
            print_card_summary(cards_in_order)
        print("\n=== Process Complete ===")
        print("The collection scanner has finished processing all rows.")
    except KeyboardInterrupt:
        print("\n\n=== SCRIPT INTERRUPTED ===")
        print("Processing was cancelled. Here is the summary of cards analyzed so far:")
        print("-" * 50)
        cards = cards_container[0] if cards_container[0] else interruptible_cards
        if SUMMARY:
            if cards:
                print_card_summary(cards)
            else:
                print("No cards were analyzed before interruption.")
        else:
            print("Summary printing disabled.")
        # Export partial CSV if enabled
        if CSV and cards:
            csv_data = prepare_csv_data(cards)
            if csv_data:
                write_csv(csv_data, f"Partial results exported to {OUTPUT_CSV}")
        print("\n=== PROCESSING STOPPED ===")
        return

if __name__ == "__main__":
    main()
