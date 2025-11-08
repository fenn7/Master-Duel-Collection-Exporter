# download_canonical_images.py
import os
import requests
from pathlib import Path
from time import sleep
from tqdm import tqdm

OUT_DIR = Path("canonical_images")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# YGOPRODeck API - returns {"data":[{...}]}
API = "https://db.ygoprodeck.com/api/v7/cardinfo.php"

def fetch_card_list():
    print("Fetching card metadata from YGOPRODeck...")
    r = requests.get(API, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data["data"]

def download_image(url, dst: Path):
    try:
        resp = requests.get(url, timeout=20, stream=True)
        resp.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in resp.iter_content(1024*8):
                f.write(chunk)
        return True
    except Exception as e:
        print("download error", url, e)
        return False

def main(limit=None):
    cards = fetch_card_list()
    print("Total cards metadata:", len(cards))
    count = 0
    for c in tqdm(cards[:limit] if limit else cards):
        cid = c.get("id")
        name = c.get("name").replace("/", "_").replace("\\", "_")
        # card_images is a list; use first image
        imgs = c.get("card_images", [])
        if not imgs:
            continue
        url = imgs[0].get("image_url")
        if not url:
            continue
        fname = OUT_DIR / f"{cid}_{name}.jpg"
        if fname.exists():
            count += 1
            continue
        ok = download_image(url, fname)
        if not ok:
            # small delay to be friendly to server
            sleep(0.2)
        count += 1
    print(f"Done. images saved to {OUT_DIR} (count approx {count})")

if __name__ == "__main__":
    main()  # optionally main(limit=500) while testing
