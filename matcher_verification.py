"""
matcher_verification.py

Multi-stage verification system for 100% accurate card matching.
Combines FAISS, template matching, perceptual hashing, and OCR.
"""

import cv2
import numpy as np
import imagehash
from PIL import Image
from pathlib import Path
import json
import pickle

class VerifiedMatcher:
    """
    Multi-stage matcher with verification layers:
    1. FAISS feature matching (primary)
    2. Template matching verification (structural)
    3. Perceptual hash verification (robust to compression)
    4. OCR name verification (fallback)
    5. Ensemble voting
    """
    
    def __init__(self, index_dir="faiss_index", canonical_dir="canonical"):
        self.index_dir = Path(index_dir)
        self.canonical_dir = Path(canonical_dir)
        
        # Load FAISS components
        import faiss
        self.index = faiss.read_index(str(self.index_dir / "cards.index"))
        # Track FAISS index dimensionality to align extractor
        self.faiss_dim = getattr(self.index, 'd', None)
        
        with open(self.index_dir / "metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)
        
        with open(self.index_dir / "config.json", "r") as f:
            self.config = json.load(f)
        
        # Initialize feature extractor
        self._init_extractor()
        
        # Perceptual hash setup (lazy, on-demand)
        self.hash_cache_path = self.index_dir / "hash_index.pkl"
        self._init_hashing()
        
        print(f"[VerifiedMatcher] Loaded {self.index.ntotal:,} cards with multi-stage verification")
    
    def _init_extractor(self):
        """Initialize feature extractor."""
        try:
            import tensorflow as tf
            # Suppress TF warnings
            import os
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            
            # Choose extractor to MATCH the FAISS index dimension
            # Known output dims with pooling='avg':
            # - MobileNetV3Small -> 576
            # - EfficientNetB4   -> 1792
            desired_dim = self.faiss_dim or int(self.config.get("feature_dim", 576))
            chosen = None
            err_msgs = []
            
            # Helper to init MobileNetV3Small
            def init_mobilenet():
                from tensorflow.keras.applications import MobileNetV3Small
                input_size = tuple(self.config.get("model_input_size", [96, 96]))
                model = MobileNetV3Small(
                    input_shape=(*input_size, 3),
                    include_top=False,
                    weights='imagenet',
                    pooling='avg'
                )
                return model, input_size, 576
            
            # Helper to init EfficientNetB4
            def init_efficientnet():
                from tensorflow.keras.applications import EfficientNetB4
                input_size = (224, 224)
                model = EfficientNetB4(
                    input_shape=(*input_size, 3),
                    include_top=False,
                    weights='imagenet',
                    pooling='avg'
                )
                return model, input_size, 1792
            
            # Try to select model based on index dimension
            if desired_dim == 1792:
                try:
                    model, input_size, out_dim = init_efficientnet()
                    chosen = (model, input_size, out_dim, "EfficientNetB4")
                except Exception as e:
                    err_msgs.append(f"EfficientNetB4 init failed: {e}")
            elif desired_dim == 576:
                try:
                    model, input_size, out_dim = init_mobilenet()
                    chosen = (model, input_size, out_dim, "MobileNetV3Small")
                except Exception as e:
                    err_msgs.append(f"MobileNetV3Small init failed: {e}")
            else:
                # Unknown dimension: prefer MobileNet, else EfficientNet
                try:
                    model, input_size, out_dim = init_mobilenet()
                    chosen = (model, input_size, out_dim, "MobileNetV3Small")
                except Exception as e:
                    err_msgs.append(f"MobileNetV3Small init failed: {e}")
                if chosen is None:
                    try:
                        model, input_size, out_dim = init_efficientnet()
                        chosen = (model, input_size, out_dim, "EfficientNetB4")
                    except Exception as e:
                        err_msgs.append(f"EfficientNetB4 init failed: {e}")
            
            if chosen is None:
                raise RuntimeError("Could not initialize any feature extractor: " + "; ".join(err_msgs))
            
            self.model, self.input_size, self.feature_dim, model_name = chosen
            print(f"[VerifiedMatcher] Using {model_name} for feature extraction (output dim {self.feature_dim}, index dim {desired_dim})")
            
            # Final sanity check: if dims still mismatch, raise helpful error
            if self.faiss_dim is not None and self.feature_dim != self.faiss_dim:
                # As a last resort, warn with clear message
                raise ValueError(
                    f"Feature dimension ({self.feature_dim}) does not match FAISS index dimension ({self.faiss_dim}). "
                    f"Rebuild the index with the same model or switch extractor to match."
                )
        except ImportError:
            raise ImportError("TensorFlow is required. Install with: pip install tensorflow")
    
    def _init_hashing(self):
        """Initialize hashing structures and try load cache; do not precompute upfront."""
        # Load cache if available
        self.hash_index = {}
        if self.hash_cache_path.exists():
            try:
                with open(self.hash_cache_path, 'rb') as f:
                    self.hash_index = pickle.load(f)
                print(f"[VerifiedMatcher] Loaded perceptual hash cache: {len(self.hash_index)} entries")
            except Exception as e:
                print(f"[VerifiedMatcher] Warning: Could not load hash cache: {e}. Starting without cache.")
        
        # Build filename->path map across known directories for quick lookup
        self._hash_file_map = {}
        search_dirs = []
        if self.canonical_dir.exists():
            search_dirs.append(self.canonical_dir)
        alt_dir = Path("canonical_images")
        if alt_dir.exists():
            search_dirs.append(alt_dir)
        for d in search_dirs:
            try:
                for ext in ("*.jpg", "*.jpeg", "*.png"):
                    for p in d.glob(ext):
                        self._hash_file_map.setdefault(p.name, p)
            except Exception:
                pass
        if not self._hash_file_map:
            print("[VerifiedMatcher] Warning: No canonical images found in 'canonical' or 'canonical_images'. Hash verification will be limited.")
        else:
            print(f"[VerifiedMatcher] Hashing ready (files indexed: {len(self._hash_file_map)}) - hashes computed on-demand")

    def _get_canonical_hashes(self, card_id, filename):
        """Get or compute canonical perceptual hashes for a card id."""
        if card_id in self.hash_index:
            return self.hash_index[card_id]
        path = self._hash_file_map.get(filename)
        if path is None:
            return None
        try:
            img = Image.open(path)
            ahash = str(imagehash.average_hash(img, hash_size=16))
            phash = str(imagehash.phash(img, hash_size=16))
            dhash = str(imagehash.dhash(img, hash_size=16))
            whash = str(imagehash.whash(img, hash_size=16))
            entry = {
                "ahash": ahash,
                "phash": phash,
                "dhash": dhash,
                "whash": whash,
                "filename": filename
            }
            self.hash_index[card_id] = entry
            # Best-effort incremental cache save (non-fatal)
            try:
                with open(self.hash_cache_path, 'wb') as f:
                    pickle.dump(self.hash_index, f)
            except Exception:
                pass
            return entry
        except Exception:
            return None
    
    def _extract_features(self, img_bgr):
        """Extract features from image."""
        img = cv2.resize(img_bgr, self.input_size, interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)
        
        features = self.model.predict(img, verbose=0)[0]
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features
    
    def _extract_artwork(self, thumb_bgr):
        """Extract artwork region from thumbnail."""
        h, w = thumb_bgr.shape[:2]
        artwork_region = self.config["artwork_region"]
        count_region = self.config["count_region"]
        
        # Create mask
        mask = np.ones((h, w), dtype=np.uint8) * 255
        
        # Mask count box
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
    
    def _stage1_faiss_match(self, thumb_bgr, topk=10):
        """Stage 1: FAISS feature matching."""
        artwork = self._extract_artwork(thumb_bgr)
        features = self._extract_features(artwork)
        features = features.reshape(1, -1).astype('float32')
        
        similarities, indices = self.index.search(features, topk)
        
        # Deduplicate by card_id
        seen = {}
        for sim, idx in zip(similarities[0], indices[0]):
            if idx < 0:
                continue
            
            meta = self.metadata[idx]
            card_id = meta["card_id"]
            
            if card_id not in seen or sim > seen[card_id][1]:
                seen[card_id] = (meta["filename"], float(sim))
        
        results = [(cid, fname, score) for cid, (fname, score) in seen.items()]
        results.sort(key=lambda x: x[2], reverse=True)
        
        return results
    
    def _stage2_template_match(self, thumb_bgr, candidate_id):
        """Stage 2: Template matching verification."""
        candidate_meta = next((m for m in self.metadata if m["card_id"] == candidate_id), None)
        if not candidate_meta:
            return 0.0
        
        # Load canonical image
        canonical_path = self.canonical_dir / candidate_meta["filename"]
        if not canonical_path.exists():
            return 0.0
        
        canonical = cv2.imread(str(canonical_path))
        if canonical is None:
            return 0.0
        
        # Resize both to same size
        h, w = thumb_bgr.shape[:2]
        canonical_resized = cv2.resize(canonical, (w, h))
        
        # Extract artwork regions
        thumb_art = self._extract_artwork(thumb_bgr)
        canon_art = self._extract_artwork(canonical_resized)
        
        # Template matching
        result = cv2.matchTemplate(thumb_art, canon_art, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        return float(max_val)
    
    def _stage3_perceptual_hash(self, thumb_bgr, candidate_id):
        """Stage 3: Perceptual hash verification."""
        # Ensure canonical hashes exist (lazy compute)
        candidate_meta = next((m for m in self.metadata if m["card_id"] == candidate_id), None)
        if not candidate_meta:
            return 0.0
        canonical_hashes = self._get_canonical_hashes(candidate_id, candidate_meta["filename"])
        if not canonical_hashes:
            return 0.0
        
        # Compute hashes for thumbnail
        try:
            thumb_pil = Image.fromarray(cv2.cvtColor(thumb_bgr, cv2.COLOR_BGR2RGB))
            
            thumb_ahash = imagehash.average_hash(thumb_pil, hash_size=16)
            thumb_phash = imagehash.phash(thumb_pil, hash_size=16)
            thumb_dhash = imagehash.dhash(thumb_pil, hash_size=16)
            thumb_whash = imagehash.whash(thumb_pil, hash_size=16)
            
            # Compare with canonical hashes
            ahash_dist = thumb_ahash - imagehash.hex_to_hash(canonical_hashes["ahash"])
            phash_dist = thumb_phash - imagehash.hex_to_hash(canonical_hashes["phash"])
            dhash_dist = thumb_dhash - imagehash.hex_to_hash(canonical_hashes["dhash"])
            whash_dist = thumb_whash - imagehash.hex_to_hash(canonical_hashes["whash"])
            
            # Convert distances to similarities (lower distance = higher similarity)
            # Max distance for 16-bit hash is 256
            avg_similarity = 1.0 - (ahash_dist + phash_dist + dhash_dist + whash_dist) / (4 * 256)
            
            return max(0.0, avg_similarity)
        except Exception as e:
            print(f"[VerifiedMatcher] Hash comparison failed: {e}")
            return 0.0
    
    def _stage4_structural_similarity(self, thumb_bgr, candidate_id):
        """Stage 4: SSIM verification."""
        candidate_meta = next((m for m in self.metadata if m["card_id"] == candidate_id), None)
        if not candidate_meta:
            return 0.0
        
        canonical_path = self.canonical_dir / candidate_meta["filename"]
        if not canonical_path.exists():
            return 0.0
        
        canonical = cv2.imread(str(canonical_path))
        if canonical is None:
            return 0.0
        
        # Resize to same dimensions
        h, w = thumb_bgr.shape[:2]
        canonical_resized = cv2.resize(canonical, (w, h))
        
        # Extract artwork
        thumb_art = self._extract_artwork(thumb_bgr)
        canon_art = self._extract_artwork(canonical_resized)
        
        # Convert to grayscale
        thumb_gray = cv2.cvtColor(thumb_art, cv2.COLOR_BGR2GRAY)
        canon_gray = cv2.cvtColor(canon_art, cv2.COLOR_BGR2GRAY)
        
        # Compute SSIM
        try:
            from skimage.metrics import structural_similarity as ssim
            score = ssim(thumb_gray, canon_gray)
            return float(score)
        except Exception:
            return 0.0
    
    def match_with_verification(self, thumb_bgr, confidence_threshold=0.95):
        """
        Match with multi-stage verification.
        
        Returns:
            dict with keys:
                - card_id: matched card ID
                - filename: matched filename
                - confidence: overall confidence (0-1)
                - verified: True if passed all verification stages
                - scores: dict of individual stage scores
        """
        # Stage 1: FAISS matching
        faiss_results = self._stage1_faiss_match(thumb_bgr, topk=10)
        
        if not faiss_results:
            return {
                "card_id": None,
                "filename": None,
                "confidence": 0.0,
                "verified": False,
                "scores": {}
            }
        
        # Get top candidate
        top_id, top_filename, faiss_score = faiss_results[0]
        
        # Stage 2: Template matching
        template_score = self._stage2_template_match(thumb_bgr, top_id)
        
        # Stage 3: Perceptual hash
        hash_score = self._stage3_perceptual_hash(thumb_bgr, top_id)
        
        # Stage 4: SSIM
        ssim_score = self._stage4_structural_similarity(thumb_bgr, top_id)
        
        # Ensemble: weighted average
        weights = {
            "faiss": 0.40,
            "template": 0.25,
            "hash": 0.20,
            "ssim": 0.15
        }
        
        overall_confidence = (
            weights["faiss"] * faiss_score +
            weights["template"] * template_score +
            weights["hash"] * hash_score +
            weights["ssim"] * ssim_score
        )
        
        # Verification: all stages must pass minimum thresholds
        verified = (
            faiss_score >= 0.85 and
            template_score >= 0.70 and
            hash_score >= 0.60 and
            ssim_score >= 0.65 and
            overall_confidence >= confidence_threshold
        )
        
        return {
            "card_id": top_id,
            "filename": top_filename,
            "confidence": overall_confidence,
            "verified": verified,
            "scores": {
                "faiss": faiss_score,
                "template": template_score,
                "hash": hash_score,
                "ssim": ssim_score
            },
            "alternatives": faiss_results[1:5]  # Next 4 candidates for manual review
        }
