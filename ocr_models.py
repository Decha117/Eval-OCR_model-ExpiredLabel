from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import cv2
from PIL import Image

_PADDLE_OCR_CACHE: dict[tuple[str, bool, str], object] = {}


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

    if importlib.util.find_spec("paddleocr"):
        models.extend(build_paddle_models())
    else:
        models.append(
            OCRModel(
                name="PaddleOCR (PP-OCRv4 / PP-OCRv3 Mobile TensorRT INT8)",
                predictor=None,
                error="ยังไม่ได้ติดตั้งแพ็กเกจ paddleocr",
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


def build_paddle_models() -> Iterable[OCRModel]:
    return [
        OCRModel(name="PaddleOCR PP-OCRv4", predictor=paddleocr_predictor("PP-OCRv4", use_gpu=True)),
        OCRModel(
            name="PaddleOCR PP-OCRv3 Mobile TensorRT INT8",
            predictor=paddleocr_predictor("PP-OCRv3", use_gpu=True, use_tensorrt=True, precision="int8"),
        ),
    ]


def paddleocr_predictor(
    ocr_version: str,
    use_gpu: bool = False,
    use_tensorrt: bool = False,
    precision: str = "fp32",
) -> Callable[[Image.Image], list[str]]:
    def _predict(image: Image.Image) -> list[str]:
        import inspect

        from paddleocr import PaddleOCR

        key = (ocr_version, use_gpu, use_tensorrt, precision)
        if key not in _PADDLE_OCR_CACHE:
            if "use_gpu" not in inspect.signature(PaddleOCR).parameters:
                raise RuntimeError(
                    "PaddleOCR เวอร์ชันนี้ไม่รองรับ use_gpu โปรดอัปเกรดเป็นเวอร์ชัน 2.7 ขึ้นไป"
                )
            kwargs = dict(
                use_angle_cls=True,
                lang="en",
                ocr_version=ocr_version,
                use_gpu=use_gpu,
                use_tensorrt=use_tensorrt,
                precision=precision,
                show_log=False,
            )
            _PADDLE_OCR_CACHE[key] = PaddleOCR(**kwargs)
        engine = _PADDLE_OCR_CACHE[key]
        result = engine.ocr(np.array(image), cls=True)
        if not result:
            return []
        lines = result[0] if isinstance(result, list) and len(result) == 1 else result
        texts: list[str] = []
        for line in lines:
            if len(line) > 1 and isinstance(line[1], (list, tuple)) and line[1]:
                texts.append(str(line[1][0]))
        return texts

    return _predict


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
