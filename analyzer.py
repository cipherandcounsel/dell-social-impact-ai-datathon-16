from __future__ import annotations

import base64
import csv
import io
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from PIL import Image
from openai import OpenAI

# NVIDIA GB10 VLM Inference Server Configuration
BASE_URL = "http://10.246.100.230/v1"
MODEL = "qwen3-vl-8b-instruct"
API_KEY = "usaadatathon"

# Works seamlessly in both Jupyter Notebook cells and standalone .py execution
PROJECT_ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
SERVER_DATASET_DIR = Path("/tmp/datathon/UTSA-flood-images-complete/images")
DEFAULT_IMAGES_DIR = SERVER_DATASET_DIR if SERVER_DATASET_DIR.exists() else (PROJECT_ROOT / "images")

# Output reports saved directly in the Jupyter working directory
REPORT_JSON_PATH = PROJECT_ROOT / "triage_report.json"
REPORT_CSV_PATH = PROJECT_ROOT / "triage_report.csv"

# Dynamic Social Vulnerability Multipliers mapped from VLM-detected land use
LAND_USE_PROFILES = {
    "Dense Urban / Commercial": {
        "pop_density": "Very High (~5,500/sq mi)",
        "vuln_multiplier": 1.45,
    },
    "Suburban Residential": {
        "pop_density": "High (~3,400/sq mi)",
        "vuln_multiplier": 1.30,
    },
    "Highway / Transit Corridor": {
        "pop_density": "Medium-High (Transit Artery)",
        "vuln_multiplier": 1.25,
    },
    "Rural / Agricultural": {
        "pop_density": "Low-Medium (~1,200/sq mi)",
        "vuln_multiplier": 1.15,
    },
}

DEFAULT_PROFILE = {
    "pop_density": "Medium (~2,800/sq mi)",
    "vuln_multiplier": 1.20,
}

TRIAGE_PROMPT = """Analyze this aerial image for flood severity and surrounding land use.
IMPORTANT: Distinguish normal water bodies (rivers, bayous, ponds within natural banks = SAFE) from floodwater (brown/muddy water covering roads, yards, vehicles, or buildings).

Classify severity into EXACTLY ONE category:
1. "VERY DANGEROUS": Severe flooding. Deep muddy water submerging houses, roofs, vehicles, or completely cutting off neighborhoods. Immediate life-threatening risk.
2. "DANGEROUS": Moderate flooding. Standing water on roads, flooded yards/fields, or edges of buildings, making travel hazardous, though structures are not deeply submerged.
3. "SAFE": No visible flooding. Dry roads, dry land, or normal water bodies within banks.

Respond ONLY with valid JSON (no markdown, no extra text):
{
  "status": "VERY DANGEROUS" | "DANGEROUS" | "SAFE",
  "base_severity": <integer from 0 to 70 based on visible flood extent>,
  "confidence": "High" | "Medium" | "Low",
  "land_use": "Dense Urban / Commercial" | "Suburban Residential" | "Highway / Transit Corridor" | "Rural / Agricultural",
  "water_type": "Muddy Floodwater" | "Standing Road Water" | "Normal River/Pond" | "None",
  "visible_infrastructure": "Short phrase listing visible roads, houses, buildings, vehicles, or bridges",
  "hazards": ["1-3 specific visible hazards, e.g. Submerged roofs, Cut-off highway, Stranded vehicles, None"],
  "evidence": "1-2 concise sentences describing visible water, roads, vehicles, or structures.",
  "dispatch_units": ["1-3 concrete units, e.g. Airboats / Shallow-Draft Boats, Rescue Helicopter, High-Clearance LMTV Trucks, Grid Power Shutoff, Drone Re-scan, Safe Staging Hub"],
  "citizen_alert_en": "Urgent SMS alert for residents in this sector (max 20 words).",
  "citizen_alert_es": "Spanish translation of the urgent SMS alert (max 20 words)."
}"""

COMPARE_PROMPT = """Compare Image 1 (BEFORE flood) and Image 2 (AFTER flood) of the same area.
1. Identify which specific roads, houses, or neighborhoods were dry in Image 1 and are now submerged in Image 2.
2. Estimate evacuation urgency (Critical / Moderate / Low).
Be concise, factual, and state any uncertainty (e.g., cloud cover or viewing angle differences)."""


def compute_priority_score(status: str, base_severity: int, multiplier: float) -> int:
    """Calculate a 0-100 Composite Priority Score combining visual flood severity and land-use vulnerability."""
    try:
        base = int(base_severity)
    except (ValueError, TypeError):
        base = {"VERY DANGEROUS": 65, "DANGEROUS": 40, "SAFE": 5}.get(status, 10)

    if status == "VERY DANGEROUS":
        base = max(base, 55)
    elif status == "DANGEROUS":
        base = max(min(base, 54), 30)
    else:
        base = min(base, 20)

    return min(100, round(base * multiplier))


def image_to_data_url(image_source, max_size: int = 768) -> str:
    """Resize an aerial image and encode it as a base64 JPEG data URL."""
    with Image.open(image_source) as img:
        img = img.convert("RGB")
        img.thumbnail((max_size, max_size))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=80)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def _extract_json(raw_text: str) -> dict:
    """Parse JSON from model output with a keyword fallback for robustness."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        upper = raw_text.upper()
        if "VERY DANGEROUS" in upper:
            status = "VERY DANGEROUS"
        elif "DANGEROUS" in upper:
            status = "DANGEROUS"
        else:
            status = "SAFE"
        return {
            "status": status,
            "base_severity": 60 if status == "VERY DANGEROUS" else (35 if status == "DANGEROUS" else 5),
            "confidence": "Low",
            "land_use": "Suburban Residential",
            "water_type": "Uncertain",
            "visible_infrastructure": "Unverified structures and roadways",
            "hazards": ["Human verification required"],
            "evidence": raw_text.strip(),
            "dispatch_units": ["Manual Dispatcher Review"],
            "citizen_alert_en": "Flooding reported in your area. Move to higher ground and await rescue instructions.",
            "citizen_alert_es": "Inundación reportada en su área. Muévase a un lugar más alto y espere instrucciones.",
        }


def analyze_image(image_source, filename: str = "uploaded_image.jpg") -> dict:
    """Run zero-shot VLM triage on a single image regardless of filename."""
    data_url = image_to_data_url(image_source)

    with OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=120, max_retries=1) as client:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            top_p=0.8,
            presence_penalty=0,
            max_tokens=380,
            extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
            messages=[
                {
                    "role": "system",
                    "content": "You are an emergency disaster-response AI running on NVIDIA GB10. Rely strictly on visible evidence and state uncertainty. Output valid JSON only.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": TRIAGE_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        )

    result = _extract_json(response.choices[0].message.content)

    # Derive sector profile dynamically from VLM-detected land use
    land_use = result.get("land_use", "Suburban Residential")
    profile = LAND_USE_PROFILES.get(land_use, DEFAULT_PROFILE)
    sector_id = Path(filename).stem.upper()

    result["file"] = filename
    result["district"] = f"Sector {sector_id} ({land_use})"
    result["pop_density"] = profile["pop_density"]
    result["critical_infra"] = result.get("visible_infrastructure", "Local Roads & Structures")
    result["priority_score"] = compute_priority_score(
        result.get("status", "SAFE"),
        result.get("base_severity", 10),
        profile["vuln_multiplier"],
    )
    result["needs_human_review"] = result.get("confidence") == "Low"
    return result


def ask_custom_question(image_sources, question: str) -> str:
    """Ask Qwen3-VL a custom question about one or multiple images (e.g., Before vs After)."""
    if not isinstance(image_sources, list):
        image_sources = [image_sources]

    content = [{"type": "text", "text": question}]
    for src in image_sources:
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(src, max_size=1024)}})

    with OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=120, max_retries=1) as client:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            top_p=0.8,
            presence_penalty=0,
            max_tokens=320,
            extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
            messages=[
                {"role": "system", "content": "Use visible image evidence. Be concise and state uncertainty."},
                {"role": "user", "content": content},
            ],
        )
    return response.choices[0].message.content


def _collect_images(folder: Path) -> list[Path]:
    """Collect all image files from the dataset directory (recursively if nested)."""
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    direct = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in extensions])
    if direct:
        return direct
    return sorted([p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in extensions])


def _save_reports(results: list[dict]) -> None:
    """Sort results by Priority Score and save JSON/CSV reports in the Jupyter directory."""
    ranked = sorted(results, key=lambda x: x.get("priority_score", 0), reverse=True)

    with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(ranked, f, indent=2, ensure_ascii=False)

    with open(REPORT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "file", "district", "status", "priority_score", "confidence",
            "water_type", "pop_density", "critical_infra", "hazards",
            "evidence", "dispatch_units", "citizen_alert_en", "citizen_alert_es",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in ranked:
            row = {k: (", ".join(v) if isinstance(v, list) else v) for k, v in r.items() if k in fieldnames}
            writer.writerow(row)


def _process_single_path(img_path: Path) -> dict:
    """Worker task for parallel image processing on the remote server."""
    try:
        res = analyze_image(img_path, filename=img_path.name)
        res["path"] = str(img_path)
        return res
    except Exception as e:
        return {
            "file": img_path.name,
            "path": str(img_path),
            "district": f"Sector {img_path.stem.upper()}",
            "pop_density": "N/A",
            "critical_infra": "N/A",
            "status": "ERROR",
            "priority_score": -1,
            "confidence": "Low",
            "needs_human_review": True,
            "water_type": "Error",
            "hazards": [str(e)],
            "evidence": "Failed to process image.",
            "dispatch_units": [],
            "citizen_alert_en": "",
            "citizen_alert_es": "",
        }


def process_folder(
    folder_path=DEFAULT_IMAGES_DIR,
    max_images: int | None = None,
    max_workers: int = 8,
    resume: bool = True,
    progress_callback=None,
) -> list[dict]:
    """Batch-process images in parallel with incremental checkpointing to CSV and JSON."""
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Dataset directory not found: {folder}")

    all_images = _collect_images(folder)
    if max_images and max_images > 0:
        all_images = all_images[:max_images]

    existing_map: dict[str, dict] = {}
    if resume and REPORT_JSON_PATH.exists():
        try:
            with open(REPORT_JSON_PATH, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    if item.get("status") != "ERROR" and item.get("file"):
                        existing_map[item["file"]] = item
        except Exception:
            existing_map = {}

    to_process = [p for p in all_images if p.name not in existing_map]
    results = list(existing_map.values())
    total_target = len(all_images)
    completed_count = len(results)

    if to_process:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_process_single_path, p): p for p in to_process}
            for future in as_completed(futures):
                img_path = futures[future]
                res = future.result()
                results.append(res)
                completed_count += 1

                if progress_callback:
                    progress_callback(completed_count, total_target, img_path.name)

                # Save incremental checkpoints every 25 images
                if completed_count % 25 == 0:
                    _save_reports(results)

    _save_reports(results)
    return sorted(results, key=lambda x: x.get("priority_score", 0), reverse=True)


if __name__ == "__main__":
    print(f"Scanning server dataset: {DEFAULT_IMAGES_DIR}")
    report = process_folder(
        DEFAULT_IMAGES_DIR,
        max_images=None,  # Process all 1,024 images
        max_workers=8,
        resume=True,
        progress_callback=lambda done, total, name: print(f"[{done}/{total}] Completed: {name}"),
    )
    print(f"\nDone! Saved {len(report)} ranked sectors to:\n  - {REPORT_JSON_PATH}\n  - {REPORT_CSV_PATH}")
