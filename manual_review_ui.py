"""
manual_review_ui.py

Console-based interface for reviewing uncertain card matches.
Shows card comparison and allows manual confirmation/correction.
"""

import cv2
import numpy as np
from pathlib import Path
import json
import os
import tempfile
from PIL import Image, ImageOps
from typing import Optional, Tuple

class ManualReviewUI:
    """Interactive UI for reviewing uncertain matches."""
    
    def __init__(self, canonical_dir="canonical"):
        self.canonical_dir = Path(canonical_dir)
        self.corrections = []
    
    def _save_temp_image(self, img_bgr: np.ndarray, prefix: str = "card") -> str:
        """Save image to temp file and return path."""
        temp_dir = Path(tempfile.gettempdir()) / "ygo_review"
        temp_dir.mkdir(exist_ok=True)
        
        # Convert BGR to RGB for PIL
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Save with a unique name
        temp_path = temp_dir / f"{prefix}_{os.getpid()}.png"
        img_pil.save(temp_path)
        return str(temp_path)
    
    def _display_ascii_art(self, img_bgr: np.ndarray, width: int = 80) -> None:
        """Display a simple ASCII art representation of the image."""
        try:
            from PIL import Image
            
            # Convert to grayscale and resize for ASCII art
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            
            # Calculate aspect ratio
            aspect_ratio = img_pil.width / img_pil.height
            new_height = int(width / (2 * aspect_ratio))  # 2x width for terminal character aspect ratio
            
            # Resize and convert to grayscale
            img_small = img_pil.resize((width, new_height), Image.Resampling.LANCZOS)
            img_gray = ImageOps.grayscale(img_small)
            
            # ASCII characters from dark to light
            ascii_chars = "@%#*+=-:. "
            
            # Convert to ASCII
            ascii_art = []
            for y in range(img_gray.height):
                line = []
                for x in range(img_gray.width):
                    # Get pixel brightness (0-255) and map to ASCII char
                    brightness = img_gray.getpixel((x, y))
                    char_index = int((brightness / 255) * (len(ascii_chars) - 1))
                    line.append(ascii_chars[char_index])
                ascii_art.append("".join(line))
            
            # Print the ASCII art
            print("\n".join(ascii_art))
            
        except ImportError:
            # Fallback if PIL is not available
            print("\n[Image preview not available - install Pillow for better visualization]")
    
    def _display_match_info(self, match_result: dict) -> None:
        """Display match information in the console."""
        print("\n" + "=" * 80)
        print(f"MATCH REVIEW NEEDED - Confidence: {match_result['confidence']:.2%}")
        print("=" * 80)
        print(f"Matched Card: {match_result['card_id']} - {match_result.get('filename', 'Unknown')}")
        print(f"Scores: FAISS={match_result['scores']['faiss']:.3f}, "
              f"Template={match_result['scores']['template']:.3f}, "
              f"Hash={match_result['scores']['hash']:.3f}, "
              f"SSIM={match_result['scores']['ssim']:.3f}")
        
        if match_result.get('alternatives'):
            print("\nTop Alternatives:")
            for i, alt in enumerate(match_result['alternatives'][:3], 1):
                # Unpack the alternative tuple (card_id, filename, score)
                if len(alt) == 3:
                    alt_id, alt_filename, alt_score = alt
                    alt_display = f"{alt_id} - {alt_filename}"
                else:
                    # Fallback if the structure is different
                    alt_id = alt[0] if len(alt) > 0 else 'Unknown'
                    alt_score = alt[2] if len(alt) > 2 else 0.0
                    alt_display = str(alt_id)
                    if hasattr(self, 'matcher') and hasattr(self.matcher, 'get_card_filename'):
                        alt_filename = self.matcher.get_card_filename(alt_id)
                        if alt_filename:
                            alt_display = f"{alt_id} - {alt_filename}"
                
                print(f"  {i}. {alt_display} (Score: {alt_score:.3f})")
        
        print("\nOptions:")
        print("  [Y] Accept match")
        print("  [N] Reject match")
        print("  [C] Enter correct card ID")
        print("  [Q] Quit review process")
        print("-" * 80)
    
    def review_match(self, thumb_bgr: np.ndarray, match_result: dict) -> dict:
        """
        Show card comparison and get user confirmation via console.
        
        Args:
            thumb_bgr: Captured thumbnail (BGR)
            match_result: Result from VerifiedMatcher
        
        Returns:
            dict with:
                - confirmed: bool
                - corrected_id: str (if user provided correction)
                - user_action: 'accept'|'reject'|'correct'
        """
        if match_result["verified"]:
            # Auto-accept verified matches
            return {"confirmed": True, "corrected_id": None, "user_action": "accept"}
        
        # Load canonical image
        canonical_path = self.canonical_dir / match_result["filename"]
        if not canonical_path.exists():
            print(f"Warning: Canonical image not found: {canonical_path}")
            return {"confirmed": False, "corrected_id": None, "user_action": "reject"}
        
        canonical = cv2.imread(str(canonical_path))
        
        # Save images to temp files for manual viewing if needed
        temp_thumb = self._save_temp_image(thumb_bgr, "captured")
        temp_canon = self._save_temp_image(canonical, "canonical")
        
        # Display UI
        print("\n" + "=" * 80)
        print("MANUAL REVIEW REQUIRED")
        print("=" * 80)
        
        # Display captured image
        print("\nCAPTURED CARD:")
        self._display_ascii_art(thumb_bgr)
        print(f"\nSaved to: {temp_thumb}")
        
        # Display canonical image
        print("\nMATCHED CARD:")
        self._display_ascii_art(canonical)
        print(f"\nSaved to: {temp_canon}")
        
        # Display match info and options
        self._display_match_info(match_result)
        
        # Get user input
        while True:
            try:
                choice = input("Enter your choice [Y/N/C/Q]: ").strip().upper()
                
                if choice == 'Y':
                    return {"confirmed": True, "corrected_id": None, "user_action": "accept"}
                elif choice == 'N':
                    return {"confirmed": False, "corrected_id": None, "user_action": "reject"}
                elif choice == 'C':
                    print("\nEnter correct card ID (or press Enter to skip):")
                    corrected_id = input("> ").strip()
                    if corrected_id:
                        return {"confirmed": True, "corrected_id": corrected_id, "user_action": "correct"}
                    else:
                        return {"confirmed": False, "corrected_id": None, "user_action": "reject"}
                elif choice == 'Q':
                    raise KeyboardInterrupt("User quit manual review")
                else:
                    print("Invalid choice. Please enter Y, N, C, or Q.")
            except KeyboardInterrupt:
                raise KeyboardInterrupt("User quit manual review")
    
    def save_corrections(self, output_path="corrections.json"):
        """Save user corrections for retraining."""
        with open(output_path, 'w') as f:
            json.dump(self.corrections, f, indent=2)
        print(f"Saved {len(self.corrections)} corrections to {output_path}")
