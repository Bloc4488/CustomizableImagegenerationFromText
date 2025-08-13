import os
import csv
from pathlib import Path
import torch
from PIL import Image
import clip
from aesthetic_predictor.aesthetic_predictor import AestheticPredictor

PROMPTS = [
    "A serene landscape with a peaceful lake reflecting the sunset, surrounded by mountains and lush trees.",
    "A quiet village covered in fresh snow, with warm lights glowing from the windows of wooden cottages.",
    "A dense jungle with towering ancient trees, vines hanging down, and a hidden temple partially covered by moss.",
    "A futuristic city filled with towering skyscrapers, flying cars, and bustling streets illuminated by artificial lights.",
    "A mysterious figure standing on the edge of a cliff, looking at a vast ocean with storm clouds rolling in.",
    "An astronaut floating in deep space, surrounded by distant galaxies and a glowing nebula.",
    "A vast desert with endless dunes, an abandoned caravan partially buried in the sand, and a lone traveler walking under the scorching sun.",
    "An enchanted forest with glowing mushrooms, floating wisps of light, and an ancient stone pathway leading deeper into the woods.",
    "A medieval battlefield at dawn, with banners flying, armored knights preparing for battle, and a castle in the distance.",
    "A bustling marketplace in an ancient city, filled with merchants selling exotic goods, colorful fabrics, and the sound of people haggling.",
    "A majestic cathedral interior illuminated by sunlight filtering through stained glass windows.",
    "A bustling steampunk cityscape with airships, mechanical gears, and Victorian architecture.",
    "A dark forest path illuminated only by glowing lanterns, shrouded in fog with ancient statues lining the way.",
    "An underwater scene showcasing vibrant coral reefs and exotic sea creatures.",
    "A colorful urban alley covered in vibrant graffiti art, with a street musician playing guitar."
]

def match_prompt_from_filename(filename_stem, prompts):
    filename_part = filename_stem.replace("_", " ").lower()
    best_prompt = None
    max_score = 0
    for prompt in prompts:
        prompt_snippet = prompt.lower()[:min(len(prompt), len(filename_part))]
        if prompt_snippet in filename_part and len(prompt_snippet) > max_score:
            best_prompt = prompt
            max_score = len(prompt_snippet)
        elif prompt.lower()[:20] in filename_part and 20 > max_score:
            best_prompt = prompt
            max_score = 20
    return best_prompt if best_prompt else filename_part

def compute_clip_score(image_path, prompt, clip_model, preprocess, device):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image)
        text_features = clip_model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).item()
    return similarity

def compute_aesthetic_score(image_path, preprocess, aesthetic_model, clip_model, device):
    img = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(input_tensor)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        score = aesthetic_model(image_features).cpu().item()
    return score

def get_fid_for_group(baseline_dir, test_dir):
    from pytorch_fid.fid_score import calculate_fid_given_paths
    fid_value = calculate_fid_given_paths(
        [str(baseline_dir), str(test_dir)],
        batch_size=32,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        dims=2048
    )
    return fid_value

def main(results_root="results"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    aesthetic_model = AestheticPredictor('aesthetic_predictor/sa_0_4_vit_b_32_linear.pth', device=device)

    results_root = Path(results_root)
    csv_path = results_root / "all_scores.csv"

    # baseline {(run, prompt): Path}
    baseline_images = {}
    for run_dir in sorted(results_root.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.isdigit():
            continue
        for group_dir in run_dir.iterdir():
            if group_dir.name.lower() != "baseline":
                continue
            for img_path in group_dir.glob("*.png"):
                prompt = match_prompt_from_filename(img_path.stem, PROMPTS)
                baseline_images[(run_dir.name, prompt)] = img_path

    group_folders = {}  # {(run, group): [img_paths]}
    all_rows = []

    for run_dir in sorted(results_root.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.isdigit() or run_dir.name != "4":
            continue
        for group_dir in run_dir.iterdir():
            if not group_dir.is_dir() or group_dir.name.lower() != "1":
                continue
            group_folders[(run_dir.name, group_dir.name)] = []
            for img_path in group_dir.glob("*.png"):
                group_folders[(run_dir.name, group_dir.name)].append(img_path)
                prompt = match_prompt_from_filename(img_path.stem, PROMPTS)
                baseline_img = baseline_images.get((run_dir.name, prompt))
                if not baseline_img:
                    print(f"Warning: baseline not found for run {run_dir.name}, prompt '{prompt}'")
                clip_score = compute_clip_score(img_path, prompt, clip_model, preprocess, device)
                aesthetic_score = compute_aesthetic_score(img_path, preprocess, aesthetic_model, clip_model, device)
                all_rows.append({
                    "run": run_dir.name,
                    "group": group_dir.name,
                    "image_path": str(img_path),
                    "prompt": prompt,
                    "clip_score": clip_score,
                    "aesthetic_score": aesthetic_score,
                    "baseline_image": str(baseline_img) if baseline_img else ""
                })
                print(f"{img_path} | CLIP: {clip_score:.4f} | Aesthetic: {aesthetic_score:.4f}")

    # FID for each group
    fid_rows = []
    for (run, group), img_paths in group_folders.items():
        if group.lower() != "1" or run != "4":
            continue
        baseline_dir = results_root / run / "baseline"
        test_dir = results_root / run / group
        if not baseline_dir.exists() or not any(baseline_dir.glob("*.png")):
            print(f"FID skipped: no baseline in {baseline_dir}")
            continue
        try:
            fid_value = get_fid_for_group(baseline_dir, test_dir)
            print(f"FID for run {run}, group {group}: {fid_value:.4f}")
            fid_rows.append({
                "run": run,
                "group": group,
                "fid": fid_value
            })
        except Exception as e:
            print(f"FID failed for {run}-{group}: {e}")

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["run", "group", "image_path", "prompt", "clip_score", "aesthetic_score", "baseline_image"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    with open(results_root / "fid_scores.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["run", "group", "fid"])
        writer.writeheader()
        for row in fid_rows:
            writer.writerow(row)

    print(f"\nAll scores saved to {csv_path} and fid_scores.csv")

if __name__ == "__main__":
    main()
