from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import cv2
from PIL import Image

def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


def preprocess_image(image: Image.Image) -> Image.Image:
    np_image = np.array(image)
    height = np_image.shape[0]
    np_image = np_image[height // 2 :, :]
    center = (np_image.shape[1] / 2, np_image.shape[0] / 2)
    rotation = cv2.getRotationMatrix2D(center, -3, 1.0)
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


@dataclass
class OCRModel:
    name: str
    predictor: Callable[[Image.Image], list[str]] | None
    error: str | None = None

    def predict(self, image: Image.Image) -> list[str]:
        if not self.predictor:
            return []
        preprocessed = preprocess_image(image)
        return self.predictor(preprocessed)


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
