from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from flask import Flask, render_template, request
from PIL import Image

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
        error: str | None = None
        labels = LabelInput("", "", "", "")
        if request.method == "POST":
            labels = LabelInput(
                production=request.form.get("production", "").strip(),
                time=request.form.get("time", "").strip(),
                expiry=request.form.get("expiry", "").strip(),
                code=request.form.get("code", "").strip(),
            )

            file = request.files.get("image")
            if not file or not file.filename:
                error = "กรุณาอัปโหลดรูปภาพ"
                return render_template(
                    "index.html",
                    labels=labels,
                    results=results,
                    error=error,
                )

            extension = Path(file.filename).suffix.lower()
            if extension not in ALLOWED_EXTENSIONS:
                error = "รองรับเฉพาะไฟล์ภาพ PNG, JPG, JPEG, BMP, TIFF"
                return render_template(
                    "index.html",
                    labels=labels,
                    results=results,
                    error=error,
                )

            image_path = UPLOAD_DIR / f"upload{extension}"
            file.save(image_path)

            try:
                image = Image.open(image_path).convert("RGB")
            except OSError:
                error = "ไม่สามารถอ่านไฟล์ภาพได้"
                return render_template(
                    "index.html",
                    labels=labels,
                    results=results,
                    error=error,
                )

            results = evaluate_models(labels, image)

        return render_template(
            "index.html",
            labels=labels,
            results=results,
            error=error,
        )

    return app


def evaluate_models(labels: LabelInput, image: Image.Image) -> list[dict]:
    label_values = labels.as_dict()
    normalized_labels = {key: normalize_text(value) for key, value in label_values.items()}

    results: list[dict] = []
    for model in build_models():
        result = evaluate_model(model, normalized_labels, image)
        results.append(result)

    results.sort(key=lambda item: item["accuracy"], reverse=True)
    return results


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
            "reason": model.error,
        }

    extracted_text = model.predict(image)
    normalized_text = normalize_text(" ".join(extracted_text))

    matched: dict[str, bool] = {}
    correct = 0
    total = len(labels)
    for key, value in labels.items():
        if value and value in normalized_text:
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
        "reason": None,
    }


def main() -> None:
    port = int(os.environ.get("PORT", "8000"))
    app = create_app()
    app.run(host="0.0.0.0", port=port, debug=True)


if __name__ == "__main__":
    main()
