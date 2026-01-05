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

    thresh = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(closed, -1, sharpen_kernel)

    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    background_removed = cv2.subtract(enhanced, blurred)
    cleaned = cv2.addWeighted(sharpened, 0.85, background_removed, 0.15, 0)
    cleaned_rgb = cv2.cvtColor(cleaned, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(cleaned_rgb)


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


def crop_text_regions(image: Image.Image) -> list[Image.Image]:
    if not importlib.util.find_spec("doctr"):
        return [image]

    from doctr.models import ocr_predictor

    predictor = ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn", pretrained=True)
    result = predictor([np.array(image)])
    export = result.export()

    width, height = image.size
    all_boxes: list[tuple[int, int, int, int]] = []

    for page in export.get("pages", []):
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                words = line.get("words", [])
                for word in words:
                    geometry = word.get("geometry")
                    if not geometry:
                        continue
                    (x_min, y_min), (x_max, y_max) = geometry
                    left = max(int(x_min * width), 0)
                    upper = max(int(y_min * height), 0)
                    right = min(int(x_max * width), width)
                    lower = min(int(y_max * height), height)
                    if right <= left or lower <= upper:
                        continue
                    box = (left, upper, right, lower)
                    all_boxes.append(box)

    if not all_boxes:
        return [image]

    return [image.crop(box) for box in all_boxes]


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
            processed_crop = preprocess_text_crop(crop)
            extracted.extend(self.predictor(processed_crop))
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

    if importlib.util.find_spec("doctr"):
        models.extend(build_doctr_models())
    else:
        models.append(
            OCRModel(
                name="Doctr (ทั้งหมด)",
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
        OCRModel(name="Doctr CRNN", predictor=doctr_predictor("db_resnet50", "crnn_vgg16_bn")),
        OCRModel(name="Doctr PARSeq", predictor=doctr_predictor("db_resnet50", "parseq")),
        OCRModel(name="Doctr linknet_resnet18", predictor=doctr_predictor("linknet_resnet18", "crnn_vgg16_bn")),
        OCRModel(name="Doctr linknet_resnet34", predictor=doctr_predictor("linknet_resnet34", "crnn_vgg16_bn")),
        OCRModel(name="Doctr linknet_resnet50", predictor=doctr_predictor("linknet_resnet50", "crnn_vgg16_bn")),
        OCRModel(name="Doctr db_resnet50", predictor=doctr_predictor("db_resnet50", "crnn_vgg16_bn")),
        OCRModel(name="Doctr db_mobilenet_v3_large", predictor=doctr_predictor("db_mobilenet_v3_large", "crnn_vgg16_bn")),
        OCRModel(name="Doctr fast_tiny", predictor=doctr_predictor("fast_tiny", "crnn_vgg16_bn")),
        OCRModel(name="Doctr fast_small", predictor=doctr_predictor("fast_small", "crnn_vgg16_bn")),
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
