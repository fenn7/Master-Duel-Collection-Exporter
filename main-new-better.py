#!/usr/bin/env python3
"""
Master Duel Collection Scraper - Performance Optimized

Detects card collection grid, captures thumbnails, extracts card info via OCR,
and exports results with minimal latency and computational overhead.
"""
import time
import signal
import sys
import re
import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from urllib.parse import quote
import cv2
import numpy as np
import pytesseract
import mss
import pyautogui
import pygetwindow as gw
import requests
import argparse

# Configuration
SCROLL_DELAY = 0
AFTER_SCROLL_DELAY = 0.1
CLICK_DESC_DELAY = 0
WINDOW_TITLE_KEYWORD = "masterduel"
OUTPUT_CSV = "collection_output"
DEBUG = False
SUMMARY = True
CSV = True
OUTPUT_DIR = "collection_csv"
# Template cache
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

def find_game_window(keyword: str = WINDOW_TITLE_KEYWORD) -> Optional[gw.Win32Window]:
    """Locate and activate the Master Duel game window"""
    try:
        wins = gw.getWindowsWithTitle(keyword.lower())
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
    """Capture screen region and return as BGR numpy array"""
    left, top, width, height = region
    try:
        with mss.mss() as sct:
            monitor = {"left": int(left), "top": int(top), "width": int(width), "height": int(height)}
            sct_img = sct.grab(monitor)
            return cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
    except Exception as e:
        if DEBUG:
            print(f"Screen capture failed for region {region}: {e}")
        raise

def load_template_cached(template_path: Path) -> Optional[np.ndarray]:
    """Load and cache template image to avoid repeated disk I/O"""
    path_str = str(template_path)
    if path_str in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[path_str]
    if not template_path.exists():
        return None
    template = cv2.imread(path_str)
    if template is not None:
        _TEMPLATE_CACHE[path_str] = template
    return template

def ocr_text_region(img: np.ndarray, config: str = r"--oem 3 --psm 7") -> str:
    """Perform OCR on a preprocessed image region"""
    if img is None or img.size == 0:
        return ""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    upscaled = cv2.resize(gray, (gray.shape[1] * 3, gray.shape[0] * 3), interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    try:
        result = pytesseract.image_to_string(thresh, lang="eng", config=config).strip()
        return result.replace("\n", " ").replace("\x0c", "").strip()
    except Exception:
        return ""

def ocr_description_zone_card_info(desc_zone_img: np.ndarray, row_number: int = 1,
                                   return_count_header: bool = False) -> Tuple:
    """Extract card name and count from description zone"""
    if desc_zone_img is None or desc_zone_img.size == 0:
        return ("", 1, 0) if return_count_header else ("", 1)
    h, w = desc_zone_img.shape[:2]
    name_h = int(h * 0.25)
    symbol_margin = int(w * 0.15)
    left_margin = int(w * 0.02)
    top_margin = int(name_h * 0.1)
    current_height = name_h - 2 - top_margin
    height_reduction = int(current_height * 0.3)
    width_reduction = int(current_height * 0.017)
    refined_top = top_margin + height_reduction // 2
    refined_bottom = name_h - 2 - (height_reduction - height_reduction // 2)
    refined_left = left_margin + width_reduction // 2
    refined_right = w - symbol_margin - (width_reduction - width_reduction // 2) - int(w * 0.017)
    name_region = desc_zone_img[refined_top:refined_bottom, refined_left:refined_right]
    card_name = ocr_text_region(name_region) if name_region.size > 0 else ""
    count = 1
    header_x = 0
    count_header_template = load_template_cached(Path("templates/count_header.PNG"))
    if count_header_template is not None:
        gray_desc = cv2.cvtColor(desc_zone_img, cv2.COLOR_BGR2GRAY)
        gray_header = cv2.cvtColor(count_header_template, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray_desc, gray_header, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val >= 0.6:
            header_x, header_y = max_loc
            header_h, header_w = gray_header.shape[:2]
            count_region_y = header_y + header_h + 2
            count_region_w = header_w + 20 - int((header_w + 20) * 0.43)
            count_region_h = min(30, h - count_region_y)
            if count_region_y + count_region_h <= h and header_x + count_region_w <= w:
                count_region = desc_zone_img[count_region_y:count_region_y + count_region_h,
                                            header_x:header_x + count_region_w]
                if count_region.size > 0:
                    count_gray = cv2.cvtColor(count_region, cv2.COLOR_BGR2GRAY)
                    count_upscaled = cv2.resize(count_gray, (count_gray.shape[1] * 3, count_gray.shape[0] * 3),
                                               interpolation=cv2.INTER_CUBIC)
                    for thresh_img in [
                        cv2.threshold(count_upscaled, 120, 255, cv2.THRESH_BINARY_INV)[1],
                        cv2.threshold(count_upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                    ]:
                        try:
                            txt = pytesseract.image_to_string(
                                thresh_img, 
                                config=r"--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789x"
                            ).strip()
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
    return (card_name, count, header_x) if return_count_header else (card_name, count)

def analyze_dism_area(dism_area_img: np.ndarray) -> int:
    """Extract dustable value from dismantle area using OCR"""
    if dism_area_img is None or dism_area_img.size == 0:
        return 0
    try:
        gray = cv2.cvtColor(dism_area_img, cv2.COLOR_BGR2GRAY)
        upscaled = cv2.resize(gray, (gray.shape[1] * 3, dism_area_img.shape[0] * 3),
                             interpolation=cv2.INTER_CUBIC)
        _, thresh = cv2.threshold(upscaled, 120, 255, cv2.THRESH_BINARY_INV)
        txt = pytesseract.image_to_string(
            thresh, 
            config=r"--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789"
        ).strip()
        return int(txt) if txt.isdigit() else 0
    except Exception:
        return 0

def get_card_info(canonical_name: str) -> dict:
    """Fetch card info from YGOPRODECK API"""
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
    """Extract rarity string from card info"""
    misc_info = card_info.get("misc_info", [])
    for misc in misc_info:
        if "md_rarity" in misc:
            rarity_map = {
                "Common": "N ",
                "Rare": "R ",
                "Super Rare": "SR",
                "Ultra Rare": "UR"
            }
            return rarity_map.get(misc["md_rarity"], "N ")
    return "N "

def determine_card_category(x_coord: int, desc_zone_width: float) -> str:
    """Determine card category (Basic/Glossy/Royal) based on count header position"""
    thresholds = [
        (0.735 * desc_zone_width, "Basic"),
        (0.811 * desc_zone_width, "Glossy"),
        (0.884 * desc_zone_width, "Royal")
    ]
    return min(thresholds, key=lambda t: abs(x_coord - t[0]))[1]

def detect_full_collection_area(win):
    """Detect collection area coordinates using header template matching"""
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

def detect_and_capture_description_zone(window_img: np.ndarray) -> Optional[np.ndarray]:
    """Detect card description zone using card header template"""
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
    window_h, window_w = window_img.shape[:2]
    desc_zone_x = header_x
    desc_zone_y = header_y + header_h + 5
    desc_zone_w = min(int(window_w * 0.22), header_w * 3)
    desc_zone_h = int(int(window_h * 0.16) * 1.73)
    desc_zone_w = min(desc_zone_w, window_w - desc_zone_x)
    desc_zone_h = min(desc_zone_h, window_h - desc_zone_y)
    if desc_zone_w <= 0 or desc_zone_h <= 0:
        return None
    return window_img[desc_zone_y:desc_zone_y + desc_zone_h, 
                     desc_zone_x:desc_zone_x + desc_zone_w]

def click_cards_and_extract_info_single_row(win, row_number: int = 1,
                                            collection_coords=None,
                                            card_dims=None) -> Dict[str, int]:
    """Process a single row of 6 cards and extract card information"""
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
    margin_x = int(card_width * 0.05)
    margin_y = int(card_height * 0.05)
    card_summary = {}
    for i in range(6):
        card_x = i * card_width
        final_x = max(0, card_x + margin_x)
        final_y = max(0, start_y + margin_y)
        final_w = min(area_w - final_x, card_width - 2 * margin_x)
        final_h = min(area_h - final_y, card_height - 2 * margin_y)
        if final_w <= 10 or final_h <= 10:
            continue
        click_x = left + card_area_x + card_x + card_width // 2
        click_y = top + card_area_y + start_y + card_height // 2
        try:
            pyautogui.click(click_x, click_y)
            time.sleep(CLICK_DESC_DELAY)
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
                                             original_dism_x:original_dism_x + refined_width]
                card_name, count, count_header_x = ocr_description_zone_card_info(
                    desc_zone_img, row_number=row_number, return_count_header=True
                )
                card_rarity = ""
                dustable_value = analyze_dism_area(dism_area_img)
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
    for row_num in range(1, 5):
        row_offset = int(collection_area_h * 0.20 * (row_num - 1))
        row_collection_coords = (
            collection_area_x,
            collection_area_y + row_offset,
            collection_area_w,
            collection_area_h - row_offset,
        )
        row_summary = click_cards_and_extract_info_single_row(
            win, row_number=row_num, collection_coords=row_collection_coords, card_dims=card_dims
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
                            # pyautogui.scroll(-1)
                            time.sleep(SCROLL_DELAY)
                    time.sleep(AFTER_SCROLL_DELAY)
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

def prepare_csv_data(cards_in_order) -> List:
    """Prepare CSV data from cards_in_order list"""
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
        for item in count_list:
            item[3] = card_rarity
        basic_count = glossy_count = royal_count = 0
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
        first_rarity = card_rarity.strip()
        visible_token = first_rarity + " " if first_rarity in ("N", "R") else first_rarity
        frame_type = card_info.get("frameType", "")
        card_type = "Trap" if frame_type == "trap" else "Spell" if frame_type == "spell" else "Monster"
        card_stats = ""
        if card_info.get("type") not in ["Trap Card", "Spell Card"]:
            level = card_info.get("level", "")
            attribute = card_info.get("attribute", "")
            race = card_info.get("race", "")
            atk = card_info.get("atk", "")
            def_val = card_info.get("def")
            def_str = "-" if def_val is None else str(def_val)
            if frame_type == "xyz":
                card_stats = f" Rank {level}, "
            elif frame_type == "link":
                linkval = card_info.get("linkval", "")
                card_stats = f" Link {linkval}, "
            else:
                card_stats = f" Level {level}, "
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
            "effect": card_info.get("desc", "").replace("\n", " ").replace("\r", " "),
        }
        csv_data.append(csv_row)
    return csv_data

def write_csv(csv_data, message=None):
    """Write CSV data to file with timestamp"""
    print("Saving collection to CSV file...")
    try:
        import csv
        csv_dir = Path(OUTPUT_DIR)
        csv_dir.mkdir(exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_[%H%M]")
        filename = f"{OUTPUT_CSV}_{timestamp}.csv"
        filepath = csv_dir / filename
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=[
                "Rarity", "Name", "Legacy Pack", "Copies", "Dustable",
                "Archetype", "Card Frame", "Card Type", "Card Stats", "Effect"
            ])
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

def print_card_summary(cards_in_order: List):
    """Print final card summary with color-coded output"""
    print("Finished processing cards. Preparing final summary...")
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
    _ansi_re = re.compile(r"\x1b\[[0-9;]*m")
    def strip_ansi(s: str) -> str:
        return _ansi_re.sub("", s) if isinstance(s, str) else ""
    def visible_len(s: str) -> int:
        return len(strip_ansi(s))
    def pad_right_ansi(s: str, width: int) -> str:
        v = visible_len(s)
        return s if v >= width else s + (" " * (width - v))
    def rarity_token_to_color(token: str) -> str:
        t = (token or "").strip()
        rarity_colors = {
            "N": RARITY_WHITE,
            "R": RARITY_LIGHT_BLUE,
            "SR": RARITY_GOLD,
            "UR": RARITY_DARK_BLUE
        }
        return rarity_colors.get(t, RARITY_BOLD)
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
        card_info = get_card_info(canonical_name)
        card_rarity = get_card_rarity_from_info(card_info)
        for item in count_list:
            item[3] = card_rarity
        basic_count = glossy_count = royal_count = 0
        for count, x_coord, desc_zone_width, _ in count_list:
            category = determine_card_category(x_coord, desc_zone_width)
            if category == "Basic":
                basic_count += count
            elif category == "Glossy":
                glossy_count += count
            elif category == "Royal":
                royal_count += count
        counts_str_elems = []
        for count, x_coord, desc_zone_width, rarity in count_list:
            category = determine_card_category(x_coord, desc_zone_width)
            category_colors = {
                "Basic": BASIC_COLOR,
                "Glossy": GLOSSY_COLOR,
                "Royal": ROYAL_COLOR
            }
            colored_category = f"{category_colors.get(category, '')}{category}{RESET}"
            counts_str_elems.append(f"{colored_category} x{count}")
        counts_str = ", ".join(counts_str_elems)
        visible_token = card_rarity
        rarity_color = rarity_token_to_color(card_rarity)
        colored_rarity_with_brackets = f"{rarity_color}{RARITY_BOLD}[{visible_token}]{RESET}"
        colored_name = f"{rarity_color}{RARITY_BOLD}{display_name}{RESET}"
        card_total = sum(int(count) for count, _, _, _ in count_list)
        card_info_dict = {
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
            legacy_pack_cards.append((idx, card_info_dict))
        else:
            gem_pack_cards.append((idx, card_info_dict))
    combined_total = max(1, len(gem_pack_cards) + len(legacy_pack_cards))
    idx_padding_width = len(str(combined_total))
    all_rows = gem_pack_cards + legacy_pack_cards
    max_rarity_w = max((visible_len(info["colored_rarity_with_brackets"]) for _, info in all_rows), default=3)
    max_name_w = max((visible_len(info["colored_name"]) for _, info in all_rows), default=10)
    max_counts_w = max((visible_len(info["counts_str"]) for _, info in all_rows), default=6)
    max_dust_w = max((len(str(info["dustable_value"])) for _, info in all_rows), default=1)
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
    if CSV:
        csv_data = prepare_csv_data(cards_in_order)
        if csv_data:
            write_csv(csv_data)

def signal_handler(sig, frame, cards_in_order_ref=None):
    """Handle Ctrl+C interruption and save partial results"""
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
    if CSV and cards:
        csv_data = prepare_csv_data(cards)
        if csv_data:
            write_csv(csv_data, f"Partial results exported to {OUTPUT_CSV}")
    print("\n=== PROCESSING STOPPED ===")
    sys.exit(0)

def update_interruptible_cards(cards_in_order):
    """Update global interruptible cards list for graceful interruption"""
    global interruptible_cards
    interruptible_cards = cards_in_order[:]

def main():
    """Main entry point - process Master Duel collection"""
    parser = argparse.ArgumentParser(description="Master Duel Collection Exporter")
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--no-summary', action='store_true', help='Disable summary printing')
    parser.add_argument('--output-dir', default='collection_csv', help='Output directory for CSV files')
    args = parser.parse_args()

    global DEBUG, SUMMARY, OUTPUT_DIR
    DEBUG = args.debug
    SUMMARY = not args.no_summary
    OUTPUT_DIR = args.output_dir

    print("Starting Master Duel Collection Exporter...")
    sys.stdout.flush()
    print("Processing collection visually via scrolling...")
    sys.stdout.flush()
    cards_container = [[]]
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, cards_container))
    win = find_game_window()
    if not win:
        print("Could not find Master Duel window. Please ensure the game is open and visible.")
        return
    print("Game window found. Starting card detection and analysis...")
    sys.stdout.flush()
    try:
        cards_in_order = process_full_collection_phases(win)
        cards_container[0] = cards_in_order
        if SUMMARY:
            print_card_summary(cards_in_order)
        print("\n=== Process Complete ===")
        sys.stdout.flush()
        print("The collection scanner has finished processing all rows.")
        sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n\n=== SCRIPT INTERRUPTED ===")
        sys.stdout.flush()
        print("Processing was cancelled. Here is the summary of cards analyzed so far:")
        sys.stdout.flush()
        print("-" * 50)
        sys.stdout.flush()
        cards = cards_container[0] if cards_container[0] else interruptible_cards
        if SUMMARY:
            if cards:
                print_card_summary(cards)
            else:
                print("No cards were analyzed before interruption.")
        else:
            print("Summary printing disabled.")
        if CSV and cards:
            csv_data = prepare_csv_data(cards)
            if csv_data:
                write_csv(csv_data, f"Partial results exported to {OUTPUT_CSV}")
        print("\n=== PROCESSING STOPPED ===")
        sys.stdout.flush()
        return

if __name__ == "__main__":
    main()
