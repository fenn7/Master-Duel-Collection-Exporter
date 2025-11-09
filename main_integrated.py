#!/usr/bin/env python3
"""
main_integrated.py

Integrated Master Duel collection scraper with multi-stage verification.
Uses VerifiedMatcher for 99%+ accuracy with human-in-the-loop for edge cases.
"""

import time
import json
import os
import pickle
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

# Import our verification system
try:
    from matcher_verification import VerifiedMatcher
    from manual_review_ui import ManualReviewUI
    VERIFICATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import verification modules: {e}")
    print("Falling back to basic matching. Install requirements: pip install imagehash scikit-image")
    VERIFICATION_AVAILABLE = False

# ---------- Configuration ----------
WINDOW_TITLE_KEYWORD = "masterduel"
MIN_CARD_AREA = 2500
MAX_CARD_AREA_RATIO = 0.12
CARD_ASPECT_MIN = 0.6
CARD_ASPECT_MAX = 1.6
OUTPUT_CSV = "collection_output.csv"

# Verification settings
CONFIDENCE_THRESHOLD = 0.95  # Require 95% confidence for auto-accept
ENABLE_MANUAL_REVIEW = True  # Set to False to skip manual review (not recommended)

# ---------- Utilities ----------

def find_game_window(keyword: str = WINDOW_TITLE_KEYWORD) -> Optional[gw.Win32Window]:
    """Find the Master Duel window using pygetwindow."""
    try:
        wins = gw.getWindowsWithTitle(keyword)
    except Exception:
        all_windows = gw.getAllWindows()
        wins = [w for w in all_windows if keyword.lower() in w.title.lower()]
    
    if not wins:
        print("Could not find game window. Please ensure Master Duel is open.")
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

def grab_region(region: Tuple[int,int,int,int]) -> np.ndarray:
    """Grab screen region using mss. Returns BGR image."""
    left, top, width, height = region
    
    with mss.mss() as sct:
        monitor = {"left": int(left), "top": int(top), "width": int(width), "height": int(height)}
        sct_img = sct.grab(monitor)
        arr = np.array(sct_img)
        
        if arr.ndim == 3 and arr.shape[2] == 4:
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        elif arr.ndim == 3 and arr.shape[2] == 3:
            img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = arr.copy()
        
        return img_bgr

def find_and_extract_first_row_cards(win) -> Tuple[bool, List]:
    """
    Extract first row of 6 cards from Master Duel window.
    Returns (success, list_of_card_images)
    """
    print("[find_and_extract_first_row_cards] Starting card detection...")
    
    # Get window coordinates
    left, top, width, height = win.left, win.top, win.width, win.height
    
    if left < 0 or top < 0:
        print(f"Window at negative coords ({left},{top}). Attempting to move...")
        try:
            win.moveTo(8, 8)
            time.sleep(0.35)
            left, top, width, height = win.left, win.top, win.width, win.height
        except Exception as e:
            print(f"Could not move window: {e}")
    
    # Grab full window
    try:
        full_window_img = grab_region((left, top, width, height))
    except Exception as e:
        print(f"Failed to grab window: {e}")
        return False, []
    
    # Load header template
    header_template_path = Path("templates/header.PNG")
    if not header_template_path.exists():
        print(f"Header template not found at {header_template_path}")
        return False, []
    
    header_template = cv2.imread(str(header_template_path))
    if header_template is None:
        print(f"Failed to load header template")
        return False, []
    
    # Find header
    gray_window = cv2.cvtColor(full_window_img, cv2.COLOR_BGR2GRAY)
    gray_header = cv2.cvtColor(header_template, cv2.COLOR_BGR2GRAY)
    
    res = cv2.matchTemplate(gray_window, gray_header, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    
    if max_val < 0.65:
        print(f"Header match confidence too low: {max_val:.3f}")
        return False, []
    
    header_x, header_y = max_loc
    header_h = gray_header.shape[0]
    
    # Define collection area
    collection_start_y = header_y + header_h + 10
    collection_region_img = full_window_img[collection_start_y:height, header_x:width]
    
    if collection_region_img.size == 0:
        print("Collection region is empty")
        return False, []
    
    # Detect collection edges (exclude slider)
    h_collection, w_collection = collection_region_img.shape[:2]
    collection_right_edge = w_collection - 57  # Exclude slider
    collection_left_edge = 0
    
    # Grid-based card detection (6 columns)
    w_collection_effective = collection_right_edge - collection_left_edge
    card_width = w_collection_effective // 6
    card_height = int(card_width * 1.4)
    
    card_images = []
    output_dir = Path("test_identifier")
    output_dir.mkdir(exist_ok=True)
    
    for col in range(6):
        x = collection_left_edge + col * card_width
        y = 0
        w = card_width
        h = min(h_collection, card_height)
        
        if x + w > collection_right_edge:
            w = collection_right_edge - x
        
        if w > 10 and h > 10:
            # Extract card region
            card_region = collection_region_img[y:y+h, x:x+w].copy()
            
            # Resize to standard size (70x100) and apply JPEG compression
            target_width, target_height = 70, 100
            card_resized = cv2.resize(card_region, (target_width, target_height), interpolation=cv2.INTER_AREA)
            
            # Apply JPEG compression to match training
            jpeg_quality = 50
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
            _, encoded = cv2.imencode('.jpg', card_resized, encode_param)
            card_final = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            
            # Save
            output_path = output_dir / f"card_{col+1:02d}.png"
            cv2.imwrite(str(output_path), card_final)
            
            card_images.append(card_final)
            print(f"Extracted card {col+1}: {target_width}x{target_height} pixels")
    
    print(f"Successfully extracted {len(card_images)} cards")
    return True, card_images

def process_cards_with_verification(card_images: List[np.ndarray], 
                                    matcher: 'VerifiedMatcher',
                                    reviewer: Optional['ManualReviewUI'] = None):
    """
    Process cards with multi-stage verification and optional manual review.
    
    Returns:
        dict: {
            'results': list of match results,
            'verified_count': number of auto-verified matches,
            'reviewed_count': number of manually reviewed matches,
            'rejected_count': number of rejected matches
        }
    """
    results = []
    verified_count = 0
    reviewed_count = 0
    rejected_count = 0
    
    print("\n" + "="*60)
    print("CARD IDENTIFICATION WITH VERIFICATION")
    print("="*60)
    
    for idx, card_img in enumerate(card_images, 1):
        print(f"\n--- Card {idx} ---")
        
        # Stage 1: Multi-stage verification
        match_result = matcher.match_with_verification(card_img, confidence_threshold=CONFIDENCE_THRESHOLD)
        
        if match_result["verified"]:
            # Auto-verified with high confidence
            print(f"✓ VERIFIED: {match_result['card_id']}")
            print(f"  Confidence: {match_result['confidence']:.2%}")
            print(f"  Scores: FAISS={match_result['scores']['faiss']:.3f}, "
                  f"Template={match_result['scores']['template']:.3f}, "
                  f"Hash={match_result['scores']['hash']:.3f}, "
                  f"SSIM={match_result['scores']['ssim']:.3f}")
            
            results.append({
                'card_number': idx,
                'card_id': match_result['card_id'],
                'filename': match_result['filename'],
                'confidence': match_result['confidence'],
                'verified': True,
                'manual_review': False
            })
            verified_count += 1
        
        else:
            # Uncertain match - needs review
            print(f"⚠ UNCERTAIN: {match_result['card_id']} - {match_result.get('filename', 'Unknown')} (confidence: {match_result['confidence']:.2%})")
            print(f"  Scores: FAISS={match_result['scores']['faiss']:.3f}, "
                  f"Template={match_result['scores']['template']:.3f}, "
                  f"Hash={match_result['scores']['hash']:.3f}, "
                  f"SSIM={match_result['scores']['ssim']:.3f}")
            
            if ENABLE_MANUAL_REVIEW and reviewer is not None:
                print(f"  → Requesting manual review...")
                
                try:
                    review_result = reviewer.review_match(card_img, match_result)
                    
                    if review_result["confirmed"]:
                        final_id = review_result.get("corrected_id") or match_result['card_id']
                        filename = match_result.get('filename', 'Unknown')
                        print(f"  ✓ User confirmed: {final_id} - {filename}")
                        
                        results.append({
                            'card_number': idx,
                            'card_id': final_id,
                            'filename': match_result['filename'],
                            'confidence': match_result['confidence'],
                            'verified': False,
                            'manual_review': True,
                            'user_corrected': review_result.get("corrected_id") is not None
                        })
                        reviewed_count += 1
                    else:
                        print(f"  ✗ User rejected match")
                        results.append({
                            'card_number': idx,
                            'card_id': None,
                            'filename': None,
                            'confidence': 0.0,
                            'verified': False,
                            'manual_review': True,
                            'rejected': True
                        })
                        rejected_count += 1
                
                except KeyboardInterrupt:
                    print("\n\nManual review interrupted by user")
                    break
            else:
                # Manual review disabled - accept with warning
                print(f"  ⚠ Auto-accepting uncertain match (manual review disabled)")
                results.append({
                    'card_number': idx,
                    'card_id': match_result['card_id'],
                    'filename': match_result['filename'],
                    'confidence': match_result['confidence'],
                    'verified': False,
                    'manual_review': False,
                    'uncertain': True
                })
    
    return {
        'results': results,
        'verified_count': verified_count,
        'reviewed_count': reviewed_count,
        'rejected_count': rejected_count
    }

def print_summary(processing_result: dict):
    """Print a summary of the processing results."""
    results = processing_result['results']
    verified_count = processing_result['verified_count']
    reviewed_count = processing_result['reviewed_count']
    rejected_count = processing_result['rejected_count']
    
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)
    print(f"Total cards processed: {len(results)}")
    print(f"Auto-verified (high confidence): {verified_count}")
    print(f"Manually reviewed: {reviewed_count}")
    print(f"Rejected: {rejected_count}")
    print()
    
    # Aggregate by card
    card_counts = {}
    for result in results:
        if result.get('card_id') and not result.get('rejected'):
            card_id = result['card_id']
            if card_id in card_counts:
                card_counts[card_id]['count'] += 1
            else:
                card_counts[card_id] = {
                    'filename': result.get('filename', 'Unknown'),
                    'count': 1
                }
    
    if card_counts:
        print("CARD COLLECTION:")
        print("-" * 60)
        for card_id, info in sorted(card_counts.items()):
            print(f"{card_id} - {info['filename']} x{info['count']}")
    
    print("="*60)

def export_results_to_csv(processing_result: dict, output_path: str = "identified_cards.csv"):
    """Export results to CSV."""
    results = processing_result['results']
    
    rows = []
    for result in results:
        rows.append({
            'card_number': result['card_number'],
            'card_id': result.get('card_id', ''),
            'filename': result.get('filename', ''),
            'confidence': result.get('confidence', 0.0),
            'verified': result.get('verified', False),
            'manual_review': result.get('manual_review', False),
            'user_corrected': result.get('user_corrected', False),
            'rejected': result.get('rejected', False),
            'uncertain': result.get('uncertain', False)
        })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\nResults exported to: {output_path}")

def main():
    """Main entry point."""
    print("="*60)
    print("Master Duel Collection Scraper")
    print("Multi-Stage Verification System")
    print("="*60)
    print()
    
    # Check if verification system is available
    if not VERIFICATION_AVAILABLE:
        print("ERROR: Verification system not available.")
        print("Please install required packages:")
        print("  pip install imagehash scikit-image tensorflow")
        return
    
    # Initialize verification system
    print("Initializing verification system...")
    try:
        matcher = VerifiedMatcher(
            index_dir="faiss_index",
            canonical_dir="canonical_images"
        )
    except Exception as e:
        print(f"ERROR: Could not initialize VerifiedMatcher: {e}")
        print("\nMake sure you have:")
        print("  1. Built the FAISS index (run build_faiss_v2.py)")
        print("  2. Canonical images in ./canonical_images/ directory")
        return
    
    # Initialize manual review UI
    reviewer = None
    if ENABLE_MANUAL_REVIEW:
        try:
            reviewer = ManualReviewUI(canonical_dir="canonical_images")
            print("Manual review UI initialized")
        except Exception as e:
            print(f"Warning: Could not initialize manual review UI: {e}")
            print("Continuing without manual review...")
    
    # Find game window
    print("\nSearching for Master Duel window...")
    win = find_game_window(WINDOW_TITLE_KEYWORD)
    if not win:
        print("ERROR: Could not find Master Duel window")
        return
    
    print(f"Found window: {win.title}")
    print(f"Position: ({win.left}, {win.top}), Size: {win.width}x{win.height}")
    
    # Extract first row of cards
    print("\nExtracting first row of cards...")
    success, card_images = find_and_extract_first_row_cards(win)
    
    if not success or not card_images:
        print("ERROR: Failed to extract cards")
        return
    
    print(f"\nSuccessfully extracted {len(card_images)} cards")
    
    # Process cards with verification
    print("\nProcessing cards with multi-stage verification...")
    processing_result = process_cards_with_verification(card_images, matcher, reviewer)
    
    # Print summary
    print_summary(processing_result)
    
    # Export to CSV
    export_results_to_csv(processing_result)
    
    # Save corrections if any were made
    if reviewer and reviewer.corrections:
        reviewer.save_corrections("corrections.json")
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
