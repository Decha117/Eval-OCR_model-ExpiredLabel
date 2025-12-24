from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
from PIL import Image


def normalize_text(text: str) -> str:
    return " ".join(text.lower().split())


@dataclass
class OCRModel:
    name: str
    predictor: Callable[[Image.Image], list[str]] | None
    error: str | None = None

    def predict(self, image: Image.Image) -> list[str]:
        if not self.predictor:
            return []
        return self.predictor(image)


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
