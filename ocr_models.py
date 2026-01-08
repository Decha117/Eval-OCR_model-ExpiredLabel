from __future__ import annotations

import importlib.util
from dataclasses import dataclass
import re
from typing import Callable, Iterable

import cv2
import numpy as np
from PIL import Image

def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


PREPROCESSING_STEPS = ("crop_bottom_half", "rotate_minus_3", "clahe")
CODE_PREFIX_MAP = {
    "-": "B",
    "_": "B",
    "8": "B",
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "6": "G",
    "7": "T",
    "9": "G",
}
ALPHA_PREFIX_MAP = {
    "U": "B",
}


def postprocess_ocr_text(text: str, production_text: str | None = None) -> str:
    if not text:
        return text

    def normalize_code(match: re.Match[str]) -> str:
        prefix = match.group(1)
        digits = match.group(2)
        if prefix.isalpha():
            normalized_prefix = ALPHA_PREFIX_MAP.get(prefix.upper(), prefix.upper())
        else:
            normalized_prefix = CODE_PREFIX_MAP.get(prefix)
            if not normalized_prefix:
                return match.group(0)
        return f"{normalized_prefix} {digits.zfill(2)}"

    def extract_production_month_year(value: str | None) -> tuple[str, str] | None:
        if not value:
            return None
        date_match = re.search(r"\b(\d{2})[/-](\d{2})[/-](\d{2,4})\b", value)
        if date_match:
            _, month, year = date_match.groups()
            return month, year[-2:]
        short_match = re.search(r"\b(\d{2})[/-](\d{2,4})\b", value)
        if short_match:
            month, year = short_match.groups()
            return month, year[-2:]
        return None

    production_month_year = extract_production_month_year(production_text)

    def normalize_expiry(match: re.Match[str]) -> str:
        day, month, year_digit = match.groups()
        if production_month_year and month == production_month_year[0]:
            year = production_month_year[1]
        else:
            year = year_digit.zfill(2)
        return f"{day}/{month}/{year}"

    normalized = re.sub(r"\b(\d{1,2})\s*:\s*(\d{2})\b", r"\1:\2", text)
    normalized = re.sub(r"\b([A-Za-z0-9\-_])\s*([0-9]{1,2})\b", normalize_code, normalized)
    normalized = re.sub(r"\b(\d{2})[/-](\d{2})[/-](\d)\b", normalize_expiry, normalized)
    return normalized


def preprocess_image(image: Image.Image, steps: Iterable[str] | None = None) -> Image.Image:
    selected = list(steps) if steps is not None else list(PREPROCESSING_STEPS)
    if not selected:
        return image.copy()

    np_image = np.array(image)

    if "crop_bottom_half" in selected:
        height = np_image.shape[0]
        np_image = np_image[height // 2 :, :]

    if "rotate_minus_3" in selected:
        center = (np_image.shape[1] / 2, np_image.shape[0] / 2)
        rotation = cv2.getRotationMatrix2D(center, -3, 1.0)
        np_image = cv2.warpAffine(
            np_image,
            rotation,
            (np_image.shape[1], np_image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    if "clahe" in selected:
        gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        np_image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)

    return Image.fromarray(np_image)


@dataclass
class OCRPrediction:
    text: list[str]
    boxes: list[tuple[int, int, int, int]]
    image: Image.Image


@dataclass
class OCRModel:
    name: str
    predictor: Callable[[Image.Image], OCRPrediction] | None
    error: str | None = None

    def predict(self, image: Image.Image, preprocess_steps: Iterable[str] | None = None) -> OCRPrediction:
        preprocessed = preprocess_image(image, preprocess_steps)
        if not self.predictor:
            return OCRPrediction(text=[], boxes=[], image=preprocessed)
        return self.predictor(preprocessed)


def build_models() -> Iterable[OCRModel]:
    models: list[OCRModel] = []

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


def build_doctr_models() -> Iterable[OCRModel]:
    try:
        import torch
    except ModuleNotFoundError:
        return [
            OCRModel(
                name="Doctr linknet_resnet34",
                predictor=None,
                error="ยังไม่ได้ติดตั้งแพ็กเกจ torch",
            ),
            OCRModel(
                name="Doctr fast_base",
                predictor=None,
                error="ยังไม่ได้ติดตั้งแพ็กเกจ torch",
            ),
        ]

    return [
        OCRModel(
            name="Doctr linknet_resnet34",
            predictor=doctr_predictor("linknet_resnet34", "crnn_vgg16_bn"),
        ),
        OCRModel(name="Doctr fast_base", predictor=doctr_predictor("fast_base", "crnn_vgg16_bn")),
    ]


def doctr_predictor(det_arch: str, reco_arch: str) -> Callable[[Image.Image], OCRPrediction]:
    def _predict(image: Image.Image) -> OCRPrediction:
        from doctr.models import ocr_predictor

        predictor = ocr_predictor(
            det_arch=det_arch,
            reco_arch=reco_arch,
            pretrained=True,
        )
        np_image = np.array(image)
        result = predictor([np_image])
        height, width = np_image.shape[:2]
        text, boxes = extract_doctr_prediction(result, width, height)
        return OCRPrediction(text=text, boxes=boxes, image=image)

    return _predict


def extract_doctr_prediction(result, width: int, height: int) -> tuple[list[str], list[tuple[int, int, int, int]]]:
    words: list[str] = []
    boxes: list[tuple[int, int, int, int]] = []
    export = result.export()
    for page in export.get("pages", []):
        for block in page.get("blocks", []):
            for line in block.get("lines", []):
                for word in line.get("words", []):
                    value = word.get("value")
                    geometry = word.get("geometry")
                    if value:
                        words.append(value)
                    if geometry and len(geometry) == 2:
                        (x_min, y_min), (x_max, y_max) = geometry
                        x1 = max(0, min(int(x_min * width), width - 1))
                        y1 = max(0, min(int(y_min * height), height - 1))
                        x2 = max(0, min(int(x_max * width), width - 1))
                        y2 = max(0, min(int(y_max * height), height - 1))
                        if x2 > x1 and y2 > y1:
                            boxes.append((x1, y1, x2, y2))
    return words, boxes
