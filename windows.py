# list_windows.py
import pygetwindow as gw
import sys
import re

def normalize_title(s: str) -> str:
    return re.sub(r'[^0-9a-z]', '', (s or "").lower())

def list_windows():
    try:
        wins = gw.getAllWindows()
    except Exception as e:
        print("pygetwindow.getAllWindows() failed:", e)
        try:
            # fallback to getWindowsWithTitle('') which sometimes works differently
            wins = gw.getWindowsWithTitle('')
        except Exception as e2:
            print("fallback failed:", e2)
            sys.exit(1)

    print(f"Found {len(wins)} windows. Showing up to 200:")
    for i, w in enumerate(wins[:200]):
        try:
            title = normalize_title(w.title)
            print(f"{i:03d}: repr(title)={repr(title)}  ; visible={w.visible} ; left,top={w.left},{w.top} ; size={w.width}x{w.height}")
        except Exception as e:
            print(f"{i:03d}: error reading window: {e}")

if __name__ == '__main__':
    list_windows()
