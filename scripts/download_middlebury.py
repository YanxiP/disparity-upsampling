"""Download the nine aligned full-size Middlebury 2005/2006 scene pairs."""
from __future__ import annotations

import hashlib
import json
import zipfile
import time
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from experiment.paths import RESULTS_ROOT
SCENES = {
    "Baby": (2006, "Baby1"),
    "Books": (2005, "Books"),
    "Bowling": (2006, "Bowling1"),
    "Cloth": (2006, "Cloth1"),
    "Dolls": (2005, "Dolls"),
    "Laundry": (2005, "Laundry"),
    "Moebius": (2005, "Moebius"),
    "Reindeer": (2005, "Reindeer"),
    "Wood": (2006, "Wood1"),
}


def main() -> None:
    out = RESULTS_ROOT / "dataset" / "middlebury"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for label, (year, scene) in SCENES.items():
        url = f"https://vision.middlebury.edu/stereo/data/scenes{year}/FullSize/zip-2views/{scene}-2views.zip"
        archive = out / f"{scene}-2views.zip"
        if not archive.exists():
            print("Downloading", label, url, flush=True)
            for attempt in range(5):
                try:
                    response = requests.get(url, timeout=120)
                    response.raise_for_status()
                    break
                except requests.RequestException:
                    if attempt == 4:
                        raise
                    time.sleep(2 * (attempt + 1))
            archive.write_bytes(response.content)
        sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
        with zipfile.ZipFile(archive) as zf:
            for member in (f"{scene}/view1.png", f"{scene}/disp1.png", f"{scene}/dmin.txt"):
                if member not in zf.namelist():
                    raise RuntimeError(f"Missing {member} in {archive}")
                target = out / member
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(zf.read(member))
        manifest[label] = {"year": year, "scene": scene, "url": url, "zip_sha256": sha256,
                           "rgb": f"{scene}/view1.png", "disparity": f"{scene}/disp1.png"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Ready:", len(manifest), "official scenes")


if __name__ == "__main__":
    main()
