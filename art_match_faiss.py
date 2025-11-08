# art_match_faiss.py
"""
Flexible FAISS matcher for the `faiss_index/` folder layout.

Expected layout:
  faiss_index/
    - cards.index         # the FAISS index (binary) - name may vary but default used
    - card_lookup.json    # index -> metadata mapping (list or dict)
    - config.json         # optional: { "metric": "ip"|"l2", "normalize": true/false, "dim": 2048 }
    - metadata.pkl        # optional python pickle with additional metadata

The module exposes:
  - match_thumbnail_bgr(thumb_bgr, topk=4) -> list of tuples:
       (card_id, filename_or_label, score, meta_dict)
    score semantics:
      - if index metric is inner-product: higher is better (cosine-like if vectors were normalized)
      - if index metric is L2: lower is better (these are squared L2 distances returned by FAISS)
"""

import os
import json
import pickle
from pathlib import Path
import numpy as np
import faiss
import cv2
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.models import resnet50
import re

# ---------------- configuration ----------------
FAISS_DIR = Path("faiss_index")
INDEX_FILE = FAISS_DIR / "cards.index"        # change if your index filename differs
LOOKUP_JSON = FAISS_DIR / "card_lookup.json"
CONFIG_JSON = FAISS_DIR / "config.json"
META_PKL = FAISS_DIR / "metadata.pkl"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOPK_DEFAULT = 4

# ---------------- model + preproc ----------------
_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(device=DEVICE):
    model = resnet50(pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    return model

_model = load_model()

def compute_embedding_from_bgr(bgr_img):
    """Compute a single L2 vector (float32) from an OpenCV BGR image."""
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    x = _transform(pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        v = _model(x).cpu().numpy().astype("float32").reshape(-1)
    return v

# ---------------- load index, lookup, config, meta ----------------
if not INDEX_FILE.exists():
    raise FileNotFoundError(f"FAISS index file not found at {INDEX_FILE} - adjust INDEX_FILE variable if needed.")

_index = faiss.read_index(str(INDEX_FILE))

# Infer metric type from index if possible
try:
    metric_type = _index.metric_type
except Exception:
    # Some FAISS wrappers may not expose metric_type; default to inner-product
    metric_type = faiss.METRIC_INNER_PRODUCT

# load config.json if present to override/guide behavior
_config = {}
if CONFIG_JSON.exists():
    try:
        with open(CONFIG_JSON, "r", encoding="utf-8") as f:
            _config = json.load(f)
    except Exception:
        _config = {}

# decide whether to normalize query vectors before search
# precedence: config.normalize (explicit) -> infer from metric_type (IP -> normalize True) -> False
_config_normalize = None
if isinstance(_config.get("normalize", None), bool):
    _config_normalize = bool(_config["normalize"])
else:
    if metric_type == faiss.METRIC_INNER_PRODUCT:
        _config_normalize = True
    else:
        _config_normalize = False

# load lookup mapping (index position -> label/filename/record)
_lookup = None
if LOOKUP_JSON.exists():
    try:
        with open(LOOKUP_JSON, "r", encoding="utf-8") as f:
            _lookup_raw = json.load(f)
        # _lookup_raw may be:
        #  - a list: index -> value
        #  - a dict: string(index) -> value or id->value
        if isinstance(_lookup_raw, list):
            _lookup = _lookup_raw
        elif isinstance(_lookup_raw, dict):
            # if the keys are '0','1','2' or integers as strings, convert to list
            keys = list(_lookup_raw.keys())
            all_int_keys = all(re.fullmatch(r"\d+", k) for k in keys)
            if all_int_keys:
                # build list up to max key
                maxk = max(int(k) for k in keys)
                arr = [None] * (maxk + 1)
                for k, v in _lookup_raw.items():
                    idx = int(k)
                    if idx < len(arr):
                        arr[idx] = v
                _lookup = arr
            else:
                # keep dict fallback
                _lookup = _lookup_raw
        else:
            _lookup = None
    except Exception:
        _lookup = None

# load optional metadata.pkl (free-form)
_meta = {}
if META_PKL.exists():
    try:
        with open(META_PKL, "rb") as f:
            _meta = pickle.load(f)
    except Exception:
        _meta = {}

# helper: canonicalize lookup entry for index i -> (card_id, label, meta)
def _lookup_entry_for_index(i):
    """Return (card_id, label, meta_dict) for index i. Be robust to various data shapes."""
    label = None
    extra = {}
    card_id = None

    if isinstance(_lookup, list):
        if i >= 0 and i < len(_lookup):
            label = _lookup[i]
    elif isinstance(_lookup, dict):
        # try string key
        key = str(i)
        if key in _lookup:
            label = _lookup[key]
        else:
            # maybe lookup maps filename->meta: skip
            label = None

    # if label is a dict, try to extract fields
    if isinstance(label, dict):
        # common shapes: {"filename": "...", "id": "...", "name": "...", "meta": {...}}
        card_id = label.get("id") or label.get("card_id") or label.get("cardId")
        # prefer 'filename' then 'name' or 'label'
        file_like = label.get("filename") or label.get("name") or label.get("label")
        if file_like:
            label = file_like
        # extra metadata = whole dict
        extra = label if isinstance(label, dict) else label
        if isinstance(label, dict):
            label = None
    # if label is a simple string, try to parse id if prefix numeric
    if isinstance(label, str):
        m = re.match(r"^(\d+)[ _-]+(.+)$", label)
        if m:
            card_id = card_id or m.group(1)
            label = m.group(2)
    # fallback: if metadata map exists and maps index -> record
    if not label and isinstance(_meta, dict):
        # try keys 'index' or str index
        if i in _meta:
            rec = _meta[i]
            if isinstance(rec, dict):
                card_id = card_id or rec.get("id")
                label = rec.get("name") or rec.get("filename") or rec.get("label")
                extra = rec
        elif str(i) in _meta:
            rec = _meta[str(i)]
            if isinstance(rec, dict):
                card_id = card_id or rec.get("id")
                label = label or rec.get("name") or rec.get("filename")
                extra = rec

    # ensure label is a simple string for downstream usage
    if isinstance(label, (list, dict)):
        # stringify if weird
        label = json.dumps(label, ensure_ascii=False)

    return str(card_id) if card_id is not None else None, (label if label is not None else None), (extra if isinstance(extra, dict) else {})

# ---------------- main matching function ----------------
def match_thumbnail_bgr(thumb_bgr, topk=TOPK_DEFAULT):
    """
    Match a thumbnail (OpenCV BGR numpy array) against the FAISS index.

    Returns a list of tuples:
       (card_id, label_or_filename, score, meta_dict)

    Score semantics:
      - if index metric is inner-product: score ~ inner product (higher==better). If stored vectors were normalized,
        score in [-1,1] behaves like cosine similarity (1 == perfect).
      - if index metric is L2: score is the squared L2 distance (lower==better). Many users convert to similarity with
        `sim = 1.0 / (1 + sqrt(score))` or `-score` if you prefer monotonic ordering.

    Example:
        results = match_thumbnail_bgr(img, topk=6)
        for card_id, label, score, meta in results:
            print(card_id, label, score)
    """
    # compute embedding
    v = compute_embedding_from_bgr(thumb_bgr).astype("float32")
    # If config or metric indicates normalization, make it so:
    if _config_normalize:
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm

    q = v.reshape(1, -1).astype("float32")

    # do search
    try:
        D, I = _index.search(q, topk)
    except Exception as e:
        # For some FAISS index wrappers another call may be needed (or index on GPU),
        # but surface a clear error for debugging.
        raise RuntimeError(f"FAISS search failed: {e}")

    D = np.array(D).reshape(-1).tolist()
    I = np.array(I).reshape(-1).tolist()

    results = []
    for idx, raw_score in zip(I, D):
        if idx < 0:
            continue
        card_id, label, meta = _lookup_entry_for_index(idx)
        # unify label: if label None then set to "index_{idx}"
        if label is None:
            label = f"index_{idx}"
        # depending on metric, raw_score semantics differ; return as-is but also provide a converted 'sim' if desired
        if metric_type == faiss.METRIC_INNER_PRODUCT:
            score = float(raw_score)  # higher better
        else:
            # for L2 metric, FAISS returns squared distances; lower is better
            score = float(raw_score)  # lower better
        results.append((card_id, label, score, meta))

    return results

# Optional small helper to convert L2 distances to a similarity 0..1 for easier thresholds
def l2dist_to_similarity(l2_dist, eps=1e-8):
    # l2_dist is squared distance or distance? FAISS usually returns squared L2.
    d = float(l2_dist)
    if d <= 0:
        return 1.0
    # convert to something like 1/(1+sqrt(d))
    return 1.0 / (1.0 + np.sqrt(d + eps))

# ---------------- simple CLI test (optional) ----------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python art_match_faiss.py <imagefile>")
        raise SystemExit(1)
    p = sys.argv[1]
    img = cv2.imread(p)
    if img is None:
        raise SystemExit(f"Failed to open {p}")
    res = match_thumbnail_bgr(img, topk=6)
    for cid, label, score, meta in res:
        if metric_type == faiss.METRIC_INNER_PRODUCT:
            print(f"{cid}\t{label}\tscore(ip)={score:.4f}\tmeta={meta}")
        else:
            print(f"{cid}\t{label}\tdist(l2)={score:.4f}\tsim~{l2dist_to_similarity(score):.4f}\tmeta={meta}")
