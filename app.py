from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from flask import Flask, render_template, request
from PIL import Image
from werkzeug.utils import secure_filename

from ocr_models import OCRModel, build_models, normalize_text

UPLOAD_DIR = Path("uploads")
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


@dataclass
class LabelInput:
    production: str
    time: str
    expiry: str
    code: str

    def as_dict(self) -> dict[str, str]:
        return {
            "Production": self.production,
            "Time": self.time,
            "Expiry": self.expiry,
            "Code": self.code,
        }


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
    UPLOAD_DIR.mkdir(exist_ok=True)

    @app.route("/", methods=["GET", "POST"])
    def index():
        results: list[dict] = []
        per_image_results: list[dict] = []
        error: str | None = None
        entries: list[dict] = []
        if request.method == "POST":
            entries = parse_entries(request)

            if not entries:
                error = "กรุณาอัปโหลดรูปภาพอย่างน้อย 1 รูป"
                return render_template(
                    "index.html",
                    entries=entries,
                    per_image_results=per_image_results,
                    results=results,
                    error=error,
                )

            for entry in entries:
                file = entry["file"]
                extension = Path(file.filename).suffix.lower()
                if extension not in ALLOWED_EXTENSIONS:
                    error = "รองรับเฉพาะไฟล์ภาพ PNG, JPG, JPEG, BMP, TIFF"
                    return render_template(
                        "index.html",
                        entries=entries,
                        per_image_results=per_image_results,
                        results=results,
                        error=error,
                    )

                filename = secure_filename(file.filename) or f"upload_{entry['index']}{extension}"
                image_path = UPLOAD_DIR / f"{entry['index']}_{filename}"
                file.save(image_path)

                try:
                    image = Image.open(image_path).convert("RGB")
                except OSError:
                    error = "ไม่สามารถอ่านไฟล์ภาพได้"
                    return render_template(
                        "index.html",
                        entries=entries,
                        per_image_results=per_image_results,
                        results=results,
                        error=error,
                    )

                entry["image"] = image

            results, per_image_results = evaluate_models(entries)

        return render_template(
            "index.html",
            entries=entries,
            per_image_results=per_image_results,
            results=results,
            error=error,
        )

    return app


def evaluate_models(entries: list[dict]) -> tuple[list[dict], list[dict]]:
    models = list(build_models())
    per_image_results: list[dict] = []
    overall: dict[str, dict] = {}

    for model in models:
        overall[model.name] = {
            "name": model.name,
            "available": model.error is None,
            "accuracy": 0.0,
            "correct": 0,
            "total": 0,
            "reason": model.error,
        }

    for entry in entries:
        labels = entry["labels"]
        normalized_labels = {key: normalize_text(value) for key, value in labels.items()}
        image = entry["image"]
        image_results: list[dict] = []

        for model in models:
            result = evaluate_model(model, normalized_labels, image)
            image_results.append(result)
            overall_result = overall[model.name]
            overall_result["correct"] += result["correct"]
            overall_result["total"] += result["total"]

        per_image_results.append(
            {
                "filename": entry["filename"],
                "labels": labels,
                "results": sorted(image_results, key=lambda item: item["accuracy"], reverse=True),
            }
        )

    overall_results: list[dict] = []
    for model in models:
        summary = overall[model.name]
        total = summary["total"]
        summary["accuracy"] = summary["correct"] / total if total else 0.0
        overall_results.append(summary)

    overall_results.sort(key=lambda item: item["accuracy"], reverse=True)
    return overall_results, per_image_results


def evaluate_model(
    model: OCRModel,
    labels: dict[str, str],
    image: Image.Image,
) -> dict:
    if model.error:
        return {
            "name": model.name,
            "available": False,
            "accuracy": 0.0,
            "matched": {},
            "correct": 0,
            "total": 0,
            "reason": model.error,
        }

    extracted_text = model.predict(image)
    normalized_text = normalize_text(" ".join(extracted_text))

    matched: dict[str, bool] = {}
    correct = 0
    total = 0
    for key, value in labels.items():
        if not value:
            matched[key] = False
            continue
        total += 1
        if value in normalized_text:
            matched[key] = True
            correct += 1
        else:
            matched[key] = False

    accuracy = correct / total if total else 0.0
    return {
        "name": model.name,
        "available": True,
        "accuracy": accuracy,
        "matched": matched,
        "correct": correct,
        "total": total,
        "reason": None,
    }


def parse_entries(request) -> list[dict]:
    indices: set[int] = set()
    for key in request.files:
        if key.startswith("image_"):
            parts = key.split("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                indices.add(int(parts[1]))

    for key in request.form:
        for prefix in ("production_", "time_", "expiry_", "code_"):
            if key.startswith(prefix):
                parts = key.split("_", 1)
                if len(parts) == 2 and parts[1].isdigit():
                    indices.add(int(parts[1]))
                break

    entries: list[dict] = []
    for index in sorted(indices):
        file = request.files.get(f"image_{index}")
        if not file or not file.filename:
            continue
        labels = LabelInput(
            production=request.form.get(f"production_{index}", "").strip(),
            time=request.form.get(f"time_{index}", "").strip(),
            expiry=request.form.get(f"expiry_{index}", "").strip(),
            code=request.form.get(f"code_{index}", "").strip(),
        )
        entries.append(
            {
                "index": index,
                "filename": file.filename,
                "file": file,
                "labels": labels.as_dict(),
            }
        )

    return entries


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    app = create_app()
    app.run(host="0.0.0.0", port=port, debug=True)


if __name__ == "__main__":
    main()
