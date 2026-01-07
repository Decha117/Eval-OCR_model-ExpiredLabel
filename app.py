from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from flask import (
    Flask,
    Response,
    render_template,
    request,
    send_from_directory,
    stream_with_context,
    url_for,
)
from PIL import Image, ImageDraw
from werkzeug.utils import secure_filename

from ocr_models import OCRModel, OCRPrediction, build_models, normalize_text, preprocess_image

UPLOAD_DIR = Path("uploads")
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
JOBS: dict[str, dict] = {}


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
                entry["image_url"] = url_for("uploaded_file", filename=image_path.name)

            results, per_image_results = evaluate_models(entries)

        return render_template(
            "index.html",
            entries=entries,
            per_image_results=per_image_results,
            results=results,
            error=error,
        )

    @app.route("/start", methods=["POST"])
    def start_job():
        entries = parse_entries(request)
        if not entries:
            return {"error": "กรุณาอัปโหลดรูปภาพอย่างน้อย 1 รูป"}, 400

        prepared: list[dict] = []
        for entry in entries:
            file = entry["file"]
            extension = Path(file.filename).suffix.lower()
            if extension not in ALLOWED_EXTENSIONS:
                return {"error": "รองรับเฉพาะไฟล์ภาพ PNG, JPG, JPEG, BMP, TIFF"}, 400

            filename = secure_filename(file.filename) or f"upload_{entry['index']}{extension}"
            image_path = UPLOAD_DIR / f"{entry['index']}_{filename}"
            file.save(image_path)
            prepared.append(
                {
                    "filename": entry["filename"],
                    "image_path": str(image_path),
                    "image_url": url_for("uploaded_file", filename=image_path.name),
                    "labels": entry["labels"],
                }
            )

        job_id = uuid.uuid4().hex
        JOBS[job_id] = {
            "progress": 0,
            "logs": [],
            "done": False,
            "error": None,
            "results": [],
            "per_image_results": [],
        }
        thread = threading.Thread(target=run_job, args=(job_id, prepared), daemon=True)
        thread.start()
        return {"job_id": job_id}

    @app.route("/progress/<job_id>")
    def progress(job_id: str):
        def generate():
            last_index = 0
            while True:
                job = JOBS.get(job_id)
                if not job:
                    break
                logs = job["logs"][last_index:]
                if logs or job["done"]:
                    payload = {
                        "progress": job["progress"],
                        "logs": logs,
                        "done": job["done"],
                        "error": job["error"],
                    }
                    yield f"data: {json.dumps(payload)}\n\n"
                    last_index = len(job["logs"])
                if job["done"]:
                    break
                time.sleep(0.5)

        return Response(stream_with_context(generate()), mimetype="text/event-stream")

    @app.route("/results/<job_id>")
    def job_results(job_id: str):
        job = JOBS.get(job_id)
        if not job:
            return {"error": "ไม่พบงานประมวลผล"}, 404
        html = render_template(
            "results.html",
            results=job["results"],
            per_image_results=job["per_image_results"],
        )
        return {"html": html}

    @app.route("/uploads/<path:filename>")
    def uploaded_file(filename: str):
        return send_from_directory(UPLOAD_DIR, filename)

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
        start_time = time.perf_counter()
        processed_image_url = save_processed_preview(image)
        image_results: list[dict] = []

        for model in models:
            result = evaluate_model(model, normalized_labels, image)
            image_results.append(result)
            overall_result = overall[model.name]
            overall_result["correct"] += result["correct"]
            overall_result["total"] += result["total"]

        processing_time_seconds = time.perf_counter() - start_time
        per_image_results.append(
            {
                "filename": entry["filename"],
                "image_url": entry.get("image_url"),
                "processed_image_url": processed_image_url,
                "labels": labels,
                "processing_time_seconds": processing_time_seconds,
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
            "output": "",
            "bbox_image_url": None,
            "correct": 0,
            "total": 0,
            "reason": model.error,
        }

    prediction = model.predict(image)
    output_text = " ".join(prediction.text)
    normalized_text = normalize_text(output_text)
    bbox_image_url = save_bbox_preview(prediction, model.name)

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
        "output": output_text,
        "bbox_image_url": bbox_image_url,
        "correct": correct,
        "total": total,
        "reason": None,
    }


def build_processed_preview(image: Image.Image) -> Image.Image:
    return preprocess_image(image)


def save_processed_preview(image: Image.Image) -> str:
    processed = build_processed_preview(image)
    filename = f"processed_{uuid.uuid4().hex}.jpg"
    image_path = UPLOAD_DIR / filename
    processed.save(image_path, format="JPEG", quality=90)
    return f"/uploads/{image_path.name}"


def save_bbox_preview(prediction: OCRPrediction, model_name: str) -> str | None:
    if not prediction.boxes:
        return None
    preview = prediction.image.copy()
    draw = ImageDraw.Draw(preview)
    for box in prediction.boxes:
        draw.rectangle(box, outline="#ef4444", width=2)
    safe_name = "".join(char if char.isalnum() else "_" for char in model_name).strip("_")
    filename = f"bbox_{safe_name}_{uuid.uuid4().hex}.jpg"
    image_path = UPLOAD_DIR / filename
    preview.save(image_path, format="JPEG", quality=90)
    return f"/uploads/{image_path.name}"


def run_job(job_id: str, entries: list[dict]) -> None:
    job = JOBS[job_id]
    job["logs"].append("เริ่มประมวลผลภาพทั้งหมด")

    try:
        models = list(build_models())
        total_steps = max(len(entries) * len(models), 1)
        completed_steps = 0
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
            job["logs"].append(f"อ่านภาพ {entry['filename']}")
            try:
                image = Image.open(entry["image_path"]).convert("RGB")
            except OSError:
                job["logs"].append(f"ไม่สามารถอ่านไฟล์ภาพ {entry['filename']}")
                raise
            job["logs"].append("เตรียมภาพ (preprocess)")
            start_time = time.perf_counter()
            processed_image_url = save_processed_preview(image)

            image_results: list[dict] = []
            for model in models:
                job["logs"].append(f"ประมวลผลด้วยโมเดล {model.name}")
                result = evaluate_model(model, normalized_labels, image)
                image_results.append(result)
                overall_result = overall[model.name]
                overall_result["correct"] += result["correct"]
                overall_result["total"] += result["total"]

                completed_steps += 1
                job["progress"] = int(completed_steps / total_steps * 100)

            processing_time_seconds = time.perf_counter() - start_time
            per_image_results.append(
                {
                    "filename": entry["filename"],
                    "image_url": entry.get("image_url"),
                    "processed_image_url": processed_image_url,
                    "labels": labels,
                    "processing_time_seconds": processing_time_seconds,
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
        job["results"] = overall_results
        job["per_image_results"] = per_image_results
        job["progress"] = 100
        job["logs"].append("เสร็จสิ้นการประมวลผล")
    except Exception as exc:  # pragma: no cover - defensive for background jobs
        job["error"] = f"เกิดข้อผิดพลาด: {exc}"
        job["logs"].append(job["error"])
    finally:
        job["done"] = True


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
