"""
build_index.py

Production script to build FAISS index from canonical card images.

Run this ONCE after calibration to build your index.

Usage:
    python build_index.py

This will:
1. Load all canonical images from canonical_images/
2. Simulate game thumbnails using calibrated parameters
3. Extract features using MobileNetV3
4. Build FAISS index with augmented variants
5. Save index to faiss_index/ directory

Typical runtime: 5-15 minutes for ~11,000 cards
"""

import cv2
import numpy as np
import faiss
from pathlib import Path
import pickle
from tqdm import tqdm
import json

# ============================================================================
# CALIBRATED PARAMETERS - UPDATE THESE FROM CALIBRATION STEP
# ============================================================================

TARGET_SIZE = (70, 100)  # (width, height) from calibration
JPEG_QUALITY = 50       # From calibration
ARTWORK_REGION = (0.08, 0.12, 0.92, 0.67)  # (left, top, right, bottom) fractions
COUNT_REGION = (0.70, 0.80, 1.0, 1.0)  # Adjusted rightwards

# ============================================================================
# CONFIGURATION
# ============================================================================

CANONICAL_IMAGES_DIR = "canonical_images"
OUTPUT_DIR = "faiss_index"
USE_AUGMENTATION = True  # Add brightness/blur variants
NUM_AUGMENT_VARIANTS = 3  # Dark, bright, blur
MODEL_INPUT_SIZE = (96, 96)  # Slightly larger than thumbnail for better features

# ============================================================================
# THUMBNAIL SIMULATION
# ============================================================================

def simulate_game_thumbnail(img_bgr, target_size=TARGET_SIZE, jpeg_quality=JPEG_QUALITY):
    """Convert high-res canonical image to match game thumbnail exactly."""
    thumb = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
    _, encoded = cv2.imencode('.jpg', thumb, encode_param)
    thumb = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return thumb

def extract_artwork_region(thumb_bgr, artwork_region=ARTWORK_REGION,
                          count_region=COUNT_REGION):
    """Extract artwork region and mask out UI elements."""
    h, w = thumb_bgr.shape[:2]
    
    # Create mask
    mask = np.ones((h, w), dtype=np.uint8) * 255
    
    # Mask count box
    x1, y1 = int(w * count_region[0]), int(h * count_region[1])
    x2, y2 = int(w * count_region[2]), int(h * count_region[3])
    mask[y1:y2, x1:x2] = 0
    
    # Apply mask
    masked = cv2.bitwise_and(thumb_bgr, thumb_bgr, mask=mask)
    
    # Crop to artwork region
    x1, y1 = int(w * artwork_region[0]), int(h * artwork_region[1])
    x2, y2 = int(w * artwork_region[2]), int(h * artwork_region[3])
    artwork = masked[y1:y2, x1:x2]
    
    return artwork

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

class FeatureExtractor:
    """Lightweight CNN feature extractor."""
    
    def __init__(self, input_size=MODEL_INPUT_SIZE):
        import tensorflow as tf
        from tensorflow.keras.applications import MobileNetV3Small
        
        print("Loading MobileNetV3-Small model...")
        self.model = MobileNetV3Small(
            input_shape=(*input_size, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        self.input_size = input_size
        self.feature_dim = 576  # MobileNetV3-Small output dimension
        print(f"Model loaded. Feature dimension: {self.feature_dim}")
    
    def extract(self, img_bgr):
        """Extract normalized feature vector from image."""
        # Resize
        img = cv2.resize(img_bgr, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        # Extract features
        features = self.model.predict(img, verbose=0)[0]
        
        # L2 normalize
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return features
    
    def extract_batch(self, images_bgr):
        """Extract features from batch of images (faster)."""
        batch = []
        for img_bgr in images_bgr:
            img = cv2.resize(img_bgr, self.input_size, interpolation=cv2.INTER_LINEAR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            batch.append(img)
        
        batch = np.array(batch)
        features = self.model.predict(batch, verbose=0)
        
        # L2 normalize each feature
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / (norms + 1e-8)
        
        return features

# ============================================================================
# AUGMENTATION
# ============================================================================

def create_augmented_variants(artwork_bgr):
    """Create augmented versions for robustness."""
    variants = []
    
    # Variant 1: Darker (simulate dim lighting)
    dark = cv2.convertScaleAbs(artwork_bgr, alpha=0.85, beta=-10)
    variants.append(('dark', dark))
    
    # Variant 2: Brighter (simulate bright screen)
    bright = cv2.convertScaleAbs(artwork_bgr, alpha=1.15, beta=10)
    variants.append(('bright', bright))
    
    # Variant 3: Slight blur (simulate motion blur from scrolling)
    blur = cv2.GaussianBlur(artwork_bgr, (3, 3), 0)
    variants.append(('blur', blur))
    
    return variants

# ============================================================================
# INDEX BUILDING
# ============================================================================

def build_index():
    """Main function to build FAISS index."""
    
    print("="*60)
    print("BUILDING FAISS INDEX FOR CARD RECOGNITION")
    print("="*60)
    
    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize feature extractor
    extractor = FeatureExtractor(input_size=MODEL_INPUT_SIZE)
    
    # Find all canonical images
    canonical_dir = Path(CANONICAL_IMAGES_DIR)
    image_files = sorted(list(canonical_dir.glob("*.jpg")) + list(canonical_dir.glob("*.png")))
    
    if len(image_files) == 0:
        print(f"ERROR: No images found in {CANONICAL_IMAGES_DIR}/")
        print("Please ensure your card images are in this directory")
        return
    
    print(f"\nFound {len(image_files)} card images")
    print(f"Augmentation: {'enabled' if USE_AUGMENTATION else 'disabled'}")
    if USE_AUGMENTATION:
        total_vectors = len(image_files) * (1 + NUM_AUGMENT_VARIANTS)
        print(f"Total vectors to create: {total_vectors:,}")
    
    # Process images
    all_features = []
    metadata = []
    
    print("\nProcessing images...")
    
    for img_path in tqdm(image_files, desc="Building index"):
        # Parse card ID from filename
        card_id = img_path.stem.split("_")[0]
        
        # Load high-res image
        img_highres = cv2.imread(str(img_path))
        if img_highres is None:
            print(f"\nWarning: Could not load {img_path}")
            continue
        
        # Simulate game thumbnail
        thumb = simulate_game_thumbnail(img_highres)
        
        # Extract artwork region (with UI masking)
        artwork = extract_artwork_region(thumb)
        
        # Extract features from base image
        features = extractor.extract(artwork)
        all_features.append(features)
        metadata.append({
            "card_id": card_id,
            "filename": img_path.name,
            "variant": "base"
        })
        
        # Add augmented variants
        if USE_AUGMENTATION:
            for variant_name, variant_img in create_augmented_variants(artwork):
                variant_features = extractor.extract(variant_img)
                all_features.append(variant_features)
                metadata.append({
                    "card_id": card_id,
                    "filename": img_path.name,
                    "variant": variant_name
                })
    
    # Convert to numpy array
    all_features = np.vstack(all_features).astype('float32')
    print(f"\nFeature matrix shape: {all_features.shape}")
    print(f"Total vectors: {len(all_features):,}")
    
    # Build FAISS index
    print("\nBuilding FAISS index...")
    dimension = all_features.shape[1]
    
    # Use IndexFlatIP for perfect accuracy (since we have <100k vectors)
    index = faiss.IndexFlatIP(dimension)
    index.add(all_features)
    
    print(f"Index built successfully")
    print(f"  Index type: IndexFlatIP (exact search)")
    print(f"  Dimension: {dimension}")
    print(f"  Total vectors: {index.ntotal:,}")
    
    # Save index
    index_path = output_path / "cards.index"
    faiss.write_index(index, str(index_path))
    print(f"\nIndex saved to: {index_path}")
    
    # Save metadata
    metadata_path = output_path / "metadata.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Metadata saved to: {metadata_path}")
    
    # Save configuration
    config = {
        "target_size": TARGET_SIZE,
        "jpeg_quality": JPEG_QUALITY,
        "artwork_region": ARTWORK_REGION,
        "count_region": COUNT_REGION,
        "model_input_size": MODEL_INPUT_SIZE,
        "feature_dim": dimension,
        "num_cards": len(image_files),
        "num_vectors": index.ntotal,
        "augmentation": USE_AUGMENTATION
    }
    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to: {config_path}")
    
    # Create card ID lookup
    card_ids = {}
    for meta in metadata:
        if meta["variant"] == "base":
            card_ids[meta["card_id"]] = meta["filename"]
    
    lookup_path = output_path / "card_lookup.json"
    with open(lookup_path, "w") as f:
        json.dump(card_ids, f, indent=2)
    print(f"Card lookup saved to: {lookup_path}")
    
    print("\n" + "="*60)
    print("INDEX BUILD COMPLETE!")
    print("="*60)
    print(f"\nNext step: Test with test_matcher.py")
    
    return index, metadata, config


if __name__ == "__main__":
    import sys
    
    # Check dependencies
    try:
        import tensorflow
        import faiss
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("\nInstall with:")
        print("  pip install tensorflow faiss-cpu")
        sys.exit(1)
    
    # Check if canonical images directory exists
    if not Path(CANONICAL_IMAGES_DIR).exists():
        print(f"ERROR: Directory '{CANONICAL_IMAGES_DIR}' not found")
        print("Please create this directory and add your card images")
        sys.exit(1)
    
    # Build index
    build_index()