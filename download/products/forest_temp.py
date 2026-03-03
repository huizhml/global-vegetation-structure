import re
from pathlib import Path
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright
import requests


AMZ_PATTERN = re.compile(r"[?&]X-Amz-Algorithm=|[?&]X-Amz-Signature=")

def guess_name_from_url(u: str) -> str:
    path = urlparse(u).path
    name = path.rsplit("/", 1)[-1] or "download.tif"
    return name



def stream_download(url: str, out_path: Path, timeout=300):
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    

def download_forest_temp(url: str, out_dir: str, **kwargs):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        captured = {"url": None}

        def on_request(req):
            u = req.url
            if AMZ_PATTERN.search(u):
                captured["url"] = u

        page.on("request", on_request)
        page.goto(url, wait_until="networkidle", timeout=120_000)

        if not captured["url"]:
            raise RuntimeError("Did not observe an X-Amz-* request. The flow may not be S3 presigned, or it may require interaction/auth.")

        presigned = captured["url"]
        print("Captured presigned URL:", presigned[:120] + "...")

        name = guess_name_from_url(presigned)
        out_dir = Path(out_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_fp = out_dir / name
        stream_download(presigned, out_fp)
        
        print(f"Saved: {out_fp}")
        context.close()
        browser.close()
