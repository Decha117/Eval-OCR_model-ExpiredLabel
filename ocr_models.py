from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import cv2
from PIL import Image


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def preprocess_image(image: Image.Image) -> Image.Image:
    np_image = np.array(image)
    center = (np_image.shape[1] / 2, np_image.shape[0] / 2)
    rotation = cv2.getRotationMatrix2D(center, -5, 1.0)
    np_image = cv2.warpAffine(
        np_image,
        rotation,
        (np_image.shape[1], np_image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(enhanced_rgb)


def preprocess_text_crop(image: Image.Image) -> Image.Image:
    np_image = np.array(image)
    height, width = np_image.shape[:2]
    scale = 3 if max(height, width) < 100 else 2
    np_image = cv2.resize(np_image, (width * scale, height * scale), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)
    softened = cv2.GaussianBlur(sharpened, (3, 3), 0)
    blended = cv2.addWeighted(sharpened, 0.85, softened, 0.15, 0)
    blended_rgb = cv2.cvtColor(blended, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(blended_rgb)


def preprocess_text_crop_variants(image: Image.Image) -> list[Image.Image]:
    base = preprocess_text_crop(image)
    np_image = np.array(base)
    gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    threshold_rgb = cv2.cvtColor(closed, cv2.COLOR_GRAY2RGB)
    return [base, Image.fromarray(threshold_rgb)]


DATE_PATTERN = re.compile(r"(?<!\d)(\d{2})[/-](\d{2})[/-](\d{2})(?!\d)")
TIME_PATTERN = re.compile(r"(?<!\d)([01]\d|2[0-3]):[0-5]\d(?!\d)")
CODE_PATTERN = re.compile(r"\b[A-Z]\s?\d{2}\b")


def _has_comparison_date(text: str) -> bool:
    if "<" not in text and ">" not in text:
        return False
    return bool(DATE_PATTERN.search(text))


def _is_relevant_text(text: str) -> bool:
    lowered = text.lower()
    keywords = ("prod", "production", "exp", "expiry", "mfg", "mfd", "date", "ผลิต", "หมดอายุ")
    if any(keyword in lowered for keyword in keywords):
        return True
    if _has_comparison_date(text):
        return True
    if DATE_PATTERN.search(text):
        return True
    if TIME_PATTERN.search(text):
        return True
    if CODE_PATTERN.search(text):
        return True
    return bool(re.search(r"\d", text))


def _clean_ocr_text(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9/:.\- ]+", " ", text).strip()


def _score_ocr_text(text: str) -> int:
    cleaned = _clean_ocr_text(text)
    score = 0
    score += len(re.findall(r"[0-9]", cleaned)) * 2
    score += len(re.findall(r"[A-Za-z]", cleaned))
    if DATE_PATTERN.search(cleaned):
        score += 6
    if TIME_PATTERN.search(cleaned):
        score += 5
    if CODE_PATTERN.search(cleaned):
        score += 3
    if _is_relevant_text(cleaned):
        score += 2
    return score


def crop_text_regions(image: Image.Image) -> list[Image.Image]:
    np_image = np.array(image)
    gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = 255 - thresh

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    connected = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel, iterations=2)
    dilated = cv2.dilate(connected, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape[:2]
    min_area = max(int(width * height * 0.005), 200)
    crops: list[Image.Image] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < min_area:
            continue
        if w < 20 or h < 12:
            continue
        if w / max(h, 1) < 1.2:
            continue
        pad_x = int(w * 0.05)
        pad_y = int(h * 0.2)
        x0 = max(x - pad_x, 0)
        y0 = max(y - pad_y, 0)
        x1 = min(x + w + pad_x, width)
        y1 = min(y + h + pad_y, height)
        crop = image.crop((x0, y0, x1, y1))
        crops.append(crop)

    if not crops:
        return [image]

    crops.sort(key=lambda c: c.size[1], reverse=True)
    return crops


@dataclass
class OCRModel:
    name: str
    predictor: Callable[[Image.Image], list[str]] | None
    error: str | None = None

    def predict(self, image: Image.Image) -> list[str]:
        if not self.predictor:
            return []
        preprocessed = preprocess_image(image)
        crops = crop_text_regions(preprocessed)
        extracted: list[str] = []
        for crop in crops:
            variants = preprocess_text_crop_variants(crop)
            best_texts: list[str] = []
            best_score = -1
            for variant in variants:
                candidate_texts = self.predictor(variant)
                candidate_joined = " ".join(candidate_texts)
                candidate_score = _score_ocr_text(candidate_joined)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_texts = candidate_texts
            cleaned = [_clean_ocr_text(text) for text in best_texts if text.strip()]
            extracted.extend([text for text in cleaned if text])
        return extracted


def build_models() -> Iterable[OCRModel]:
    models: list[OCRModel] = []

    if importlib.util.find_spec("rapidocr_onnxruntime"):
        models.append(OCRModel(name="RapidOCR", predictor=rapidocr_predictor))
    else:
        models.append(
            OCRModel(
                name="RapidOCR",
                predictor=None,
                error="ยังไม่ได้ติดตั้งแพ็กเกจ rapidocr-onnxruntime",
            )
        )

    models.append(build_tesseract_model())

    if importlib.util.find_spec("doctr"):
        models.extend(build_doctr_models())
    else:
        models.append(
            OCRModel(
                name="Doctr (fast_base, linknet_resnet34)",
                predictor=None,
                error="ยังไม่ได้ติดตั้งแพ็กเกจ python-doctr",
            )
        )

    return models


def rapidocr_predictor(image: Image.Image) -> list[str]:
    from rapidocr_onnxruntime import RapidOCR

    engine = RapidOCR()
    result, _ = engine(np.array(image))
    if not result:
        return []
    return [text for _, text, _ in result]


def build_doctr_models() -> Iterable[OCRModel]:
    return [
        OCRModel(name="Doctr linknet_resnet34", predictor=doctr_predictor("linknet_resnet34", "crnn_vgg16_bn")),
        OCRModel(name="Doctr fast_base", predictor=doctr_predictor("fast_base", "crnn_vgg16_bn")),
    ]


def doctr_predictor(det_arch: str, reco_arch: str) -> Callable[[Image.Image], list[str]]:
    def _predict(image: Image.Image) -> list[str]:
        from doctr.models import ocr_predictor

        predictor = ocr_predictor(det_arch=det_arch, reco_arch=reco_arch, pretrained=True)
        np_image = np.array(image)
        result = predictor([np_image])
        return extract_doctr_text(result)

    return _predict


def extract_doctr_text(result) -> list[str]:
    words: list[str] = []
    export = result.export()
    for page in export.get("pages", []):
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                for word in line.get("words", []):
                    value = word.get("value")
                    if value:
                        words.append(value)
    return words


def build_tesseract_model() -> OCRModel:
    if not importlib.util.find_spec("pytesseract"):
        return OCRModel(
            name="Tesseract",
            predictor=None,
            error="ยังไม่ได้ติดตั้งแพ็กเกจ pytesseract",
        )

    try:
        import pytesseract

        _ = pytesseract.get_tesseract_version()
    except (ImportError, RuntimeError, OSError) as exc:
        return OCRModel(
            name="Tesseract",
            predictor=None,
            error=f"Tesseract ใช้งานไม่ได้: {exc}",
        )

    return OCRModel(name="Tesseract", predictor=tesseract_predictor)


def tesseract_predictor(image: Image.Image) -> list[str]:
    import pytesseract

    text = pytesseract.image_to_string(image, config="--oem 3 --psm 6")
    return [line.strip() for line in text.splitlines() if line.strip()]
