"""
test_matcher.py

Test the card recognition system on real game thumbnails.

Usage:
    python test_matcher.py path/to/game_thumbnail.jpg
    python test_matcher.py --batch path/to/screenshots/
    python test_matcher.py --benchmark
"""

import cv2
import numpy as np
import faiss
import pickle
import json
from pathlib import Path
import argparse
import time

# ============================================================================
# CARD MATCHER CLASS
# ============================================================================

class CardMatcher:
    """Fast card matching using FAISS index."""
    
    def __init__(self, index_dir="faiss_index"):
        """Load pre-built FAISS index and metadata."""
        print(f"Loading index from {index_dir}...")
        
        index_path = Path(index_dir)
        
        # Load config
        with open(index_path / "config.json", "r") as f:
            self.config = json.load(f)
        
        # Load FAISS index
        self.index = faiss.read_index(str(index_path / "cards.index"))
        
        # Load metadata
        with open(index_path / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)
        
        # Load feature extractor
        self._init_extractor()
        
        # OCR for copy count (lazy load)
        self.ocr = None
        
        print(f"Index loaded: {self.index.ntotal:,} vectors")
        print(f"Ready for matching!")
    
    def _init_extractor(self):
        """Initialize feature extractor with same model as training."""
        import tensorflow as tf
        from tensorflow.keras.applications import MobileNetV3Small
        
        input_size = tuple(self.config["model_input_size"])
        
        self.model = MobileNetV3Small(
            input_shape=(*input_size, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        self.input_size = input_size
    
    def _extract_features(self, img_bgr):
        """Extract features from image."""
        # Resize
        img = cv2.resize(img_bgr, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        # Extract
        features = self.model.predict(img, verbose=0)[0]
        
        # L2 normalize
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features
    
    def _extract_artwork(self, thumb_bgr):
        """Extract artwork region from thumbnail (same as training)."""
        h, w = thumb_bgr.shape[:2]
        
        artwork_region = self.config["artwork_region"]
        count_region = self.config["count_region"]
        
        # Create mask
        mask = np.ones((h, w), dtype=np.uint8) * 255
        
        # Mask count box only
        x1, y1 = int(w * count_region[0]), int(h * count_region[1])
        x2, y2 = int(w * count_region[2]), int(h * count_region[3])
        mask[y1:y2, x1:x2] = 0
        
        # Apply mask
        masked = cv2.bitwise_and(thumb_bgr, thumb_bgr, mask=mask)
        
        # Crop to artwork
        x1, y1 = int(w * artwork_region[0]), int(h * artwork_region[1])
        x2, y2 = int(w * artwork_region[2]), int(h * artwork_region[3])
        artwork = masked[y1:y2, x1:x2]
        
        return artwork
    
    def match(self, thumb_bgr, topk=5):
        """
        Match a card thumbnail to database.
        
        Args:
            thumb_bgr: Game thumbnail (66x97 or similar)
            topk: Return top-k matches
        
        Returns:
            List of (card_id, filename, confidence_score)
        """
        # Extract artwork
        artwork = self._extract_artwork(thumb_bgr)
        
        # Extract features
        features = self._extract_features(artwork)
        features = features.reshape(1, -1).astype('float32')
        
        # Search index
        similarities, indices = self.index.search(features, topk * 4)
        
        # Deduplicate by card_id
        seen = {}
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < 0:
                continue
            
            meta = self.metadata[idx]
            card_id = meta["card_id"]
            
            if card_id not in seen or sim > seen[card_id][1]:
                seen[card_id] = (meta["filename"], float(sim))
            
            if len(seen) >= topk:
                break
        
        results = [(cid, fname, score) for cid, (fname, score) in seen.items()]
        results.sort(key=lambda x: x[2], reverse=True)
        
        return results[:topk]
    
    def _init_ocr(self):
        """Lazy initialization of OCR engine."""
        if self.ocr is None:
            try:
                from paddleocr import PaddleOCR
                
                # Try different API versions of PaddleOCR
                try:
                    # Newer API (v2.7+) - no show_log parameter
                    self.ocr = PaddleOCR(
                        use_angle_cls=False,
                        lang='en',
                        use_gpu=False
                    )
                except (TypeError, ValueError):
                    # Older API - has show_log parameter
                    try:
                        self.ocr = PaddleOCR(
                            use_angle_cls=False,
                            lang='en',
                            show_log=False
                        )
                    except:
                        # If both fail, disable OCR
                        raise ImportError("Could not initialize PaddleOCR")
                    
            except ImportError:
                print("Warning: PaddleOCR not available")
                print("Copy count detection disabled.")
                self.ocr = False
            except Exception as e:
                print(f"Warning: PaddleOCR initialization failed: {e}")
                print("Copy count detection disabled.")
                self.ocr = False
    
    def extract_copy_count(self, thumb_bgr):
        """Extract copy count using OCR."""
        self._init_ocr()
        
        # Check if OCR is disabled or failed to initialize
        if self.ocr is None or self.ocr is False:
            return None
        
        h, w = thumb_bgr.shape[:2]
        count_region = self.config["count_region"]
        
        # Extract count box region
        x1, y1 = int(w * count_region[0]), int(h * count_region[1])
        x2, y2 = int(w * count_region[2]), int(h * count_region[3])
        count_box = thumb_bgr[y1:y2, x1:x2]
        
        # Enhance for OCR
        gray = cv2.cvtColor(count_box, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Run OCR
        try:
            result = self.ocr.ocr(binary, cls=False)
            if result and result[0]:
                text = result[0][0][1][0]
                import re
                numbers = re.findall(r'\d+', text)
                if numbers:
                    return int(numbers[0])
        except Exception as e:
            print(f"OCR error: {e}")
        
        return None
    
    def process_card(self, thumb_bgr):
        """Complete processing: ID + copy count."""
        matches = self.match(thumb_bgr, topk=5)
        copy_count = self.extract_copy_count(thumb_bgr)
        
        return {
            "card_id": matches[0][0] if matches else None,
            "filename": matches[0][1] if matches else None,
            "confidence": matches[0][2] if matches else 0.0,
            "copy_count": copy_count,
            "top_matches": matches
        }

# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_single_image(matcher, image_path):
    """Test on a single thumbnail."""
    print(f"\nTesting: {image_path}")
    print("-" * 60)
    
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Error: Could not load {image_path}")
        return
    
    h, w = img.shape[:2]
    print(f"Image size: {w}x{h}")
    
    # Check if image size matches expected size
    expected_size = tuple(matcher.config["target_size"])
    if (w, h) != expected_size:
        print(f"⚠️  WARNING: Image size {w}x{h} doesn't match expected {expected_size}")
        print(f"   Index was built for {expected_size[0]}x{expected_size[1]} thumbnails")
        print(f"   Results may be poor. Consider:")
        print(f"   1. Resizing image to {expected_size}, OR")
        print(f"   2. Rebuilding index with TARGET_SIZE = ({w}, {h})")
    
    # Time the matching
    start = time.time()
    result = matcher.process_card(img)
    elapsed = (time.time() - start) * 1000
    
    print(f"\n✓ Match found in {elapsed:.1f}ms")
    print(f"\nBest match:")
    print(f"  Card ID: {result['card_id']}")
    print(f"  Filename: {result['filename']}")
    print(f"  Confidence: {result['confidence']:.4f}")
    
    if result['copy_count']:
        print(f"  Copy count: {result['copy_count']}")
    
    print(f"\nTop 5 matches:")
    for i, (cid, fname, score) in enumerate(result['top_matches'], 1):
        print(f"  {i}. {cid:12s} {fname:50s} {score:.4f}")
    
    return result

def test_batch(matcher, images_dir):
    """Test on batch of images."""
    images = list(Path(images_dir).glob("*.jpg")) + list(Path(images_dir).glob("*.png"))
    
    if len(images) == 0:
        print(f"No images found in {images_dir}")
        return
    
    print(f"\nTesting {len(images)} images from {images_dir}")
    print("=" * 60)
    
    total_time = 0
    results = []
    
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        start = time.time()
        result = matcher.process_card(img)
        elapsed = time.time() - start
        total_time += elapsed
        
        # Extract card name from filename
        card_name = ""
        if result['filename']:
            # Format is usually "12345678_Card Name.jpg"
            parts = result['filename'].split('_', 1)
            if len(parts) > 1:
                card_name = parts[1].rsplit('.', 1)[0]  # Remove extension
            else:
                card_name = result['filename'].rsplit('.', 1)[0]
        
        results.append({
            "filename": img_path.name,
            "card_id": result['card_id'],
            "card_name": card_name,
            "confidence": result['confidence'],
            "time_ms": elapsed * 1000
        })
        
        print(f"{img_path.name:20s} -> {result['card_id']:12s} {card_name[:30]:30s} ({result['confidence']:.3f}) [{elapsed*1000:.1f}ms]")
    
    avg_time = (total_time / len(results)) * 1000
    print("\n" + "=" * 60)
    print(f"Processed {len(results)} cards")
    print(f"Average time: {avg_time:.1f}ms per card")
    print(f"Total time: {total_time:.2f}s")

def benchmark_performance(matcher):
    """Benchmark matching performance."""
    print("\nRunning performance benchmark...")
    print("=" * 60)
    
    # Create dummy thumbnail
    dummy = np.random.randint(0, 255, (97, 66, 3), dtype=np.uint8)
    
    # Warm-up
    for _ in range(5):
        _ = matcher.match(dummy)
    
    # Benchmark
    num_runs = 100
    start = time.time()
    for _ in range(num_runs):
        _ = matcher.match(dummy)
    elapsed = time.time() - start
    
    avg_time = (elapsed / num_runs) * 1000
    throughput = num_runs / elapsed
    
    print(f"Benchmark results ({num_runs} iterations):")
    print(f"  Average latency: {avg_time:.2f}ms")
    print(f"  Throughput: {throughput:.1f} cards/sec")
    print(f"  Total time: {elapsed:.2f}s")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test Yu-Gi-Oh card matcher")
    parser.add_argument("path", nargs="?", help="Path to thumbnail image or directory")
    parser.add_argument("--batch", action="store_true", help="Process batch of images")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--index-dir", default="faiss_index", help="Index directory")
    
    args = parser.parse_args()
    
    # Load matcher
    try:
        matcher = CardMatcher(args.index_dir)
    except Exception as e:
        print(f"Error loading index: {e}")
        print("\nMake sure you've run build_index.py first!")
        return
    
    # Run benchmark
    if args.benchmark:
        benchmark_performance(matcher)
        return
    
    # Test on provided image(s)
    if args.path:
        path = Path(args.path)
        
        if args.batch or path.is_dir():
            test_batch(matcher, path)
        elif path.is_file():
            test_single_image(matcher, path)
        else:
            print(f"Error: {path} not found")
    else:
        # Interactive mode
        print("\nNo image provided. Enter path to test image:")
        print("(or press Ctrl+C to exit)")
        
        while True:
            try:
                path = input("\nImage path: ").strip()
                if Path(path).is_file():
                    test_single_image(matcher, path)
                else:
                    print(f"File not found: {path}")
            except KeyboardInterrupt:
                print("\nExiting...")
                break

if __name__ == "__main__":
    main()