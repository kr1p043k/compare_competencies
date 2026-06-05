"""Download all PDFs from Yandex Disk for direction 09.03.02"""
import requests, os, sys, time

PUBLIC_KEY = "https://disk.360.yandex.ru/d/-5D0p0XfTwL5Qg"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "temp", "rpd_pdfs")
os.makedirs(OUT_DIR, exist_ok=True)

def list_all(path="/"):
    r = requests.get("https://cloud-api.yandex.net/v1/disk/public/resources", params={
        "public_key": PUBLIC_KEY, "path": path, "limit": 100,
    })
    return r.json().get("_embedded", {}).get("items", [])

def get_download_url(path):
    r = requests.get("https://cloud-api.yandex.net/v1/disk/public/resources/download", params={
        "public_key": PUBLIC_KEY, "path": path,
    })
    return r.json().get("href")

def download_file(url, fpath):
    r = requests.get(url, stream=True, timeout=30)
    with open(fpath, "wb") as f:
        for chunk in r.iter_content(8192):
            f.write(chunk)

def walk(path="/"):
    items = list_all(path)
    for item in items:
        name = item["name"]
        item_path = (path.rstrip("/") + "/" + name) if path != "/" else "/" + name
        if item["type"] == "dir":
            print(f"[DIR] {name}/")
            walk(item_path)
        elif name.endswith(".pdf"):
            dl_url = get_download_url(item_path)
            if dl_url:
                out = os.path.join(OUT_DIR, name)
                if os.path.exists(out):
                    print(f"  [SKIP] {name}")
                else:
                    print(f"  [DL]   {name} ({item['size']//1024} KB)")
                    download_file(dl_url, out)
                    time.sleep(0.3)

walk()
print(f"\nDone. Files in {OUT_DIR}: {len([f for f in os.listdir(OUT_DIR) if f.endswith('.pdf')])}")
