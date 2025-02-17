import os
import requests
from duckduckgo_search import DDGS
from PIL import Image
from io import BytesIO

SEARCH_QUERY = "Future"
NUM_IMAGES = 2000
OUTPUT_FOLDER = "Styles/Future"

def fetch_image_urls():
    urls = []
    with DDGS() as ddgs:
        results = ddgs.images(SEARCH_QUERY, max_results=NUM_IMAGES)
        for res in results:
            urls.append(res["image"])
    return urls

def download_images(image_urls):
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    for i, img_url in enumerate(image_urls):
        try:
            response = requests.get(img_url, stream=True, timeout=5)
            response.raise_for_status()

            img = Image.open(BytesIO(response.content))
            img_path = os.path.join(OUTPUT_FOLDER, f"vangogh_{i+1}.jpg")
            img.convert("RGB").save(img_path, "JPEG")

            print(f"Downloaded: {img_path}")
        except Exception as e:
            print(f"Failed to download {img_url}: {e}")

if __name__ == "__main__":
    urls = fetch_image_urls()
    download_images(urls)
