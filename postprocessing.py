from __future__ import annotations

import re

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


def postprocess_ocr_text(text: str) -> str:
    if not text:
        return text

    def normalize_code(match: re.Match[str]) -> str:
        prefix = match.group(1)
        digits = match.group(2)
        if prefix.isalpha():
            normalized_prefix = prefix.upper()
        else:
            normalized_prefix = CODE_PREFIX_MAP.get(prefix)
            if not normalized_prefix:
                return match.group(0)
        return f"{normalized_prefix} {digits}"

    normalized = re.sub(r"\b(\d{1,2})\s*:\s*(\d{2})\b", r"\1:\2", text)
    normalized = re.sub(r"\b([A-Za-z0-9\-_])\s*([0-9]{2})\b", normalize_code, normalized)
    return normalized
