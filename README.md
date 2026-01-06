# Eval-OCR_model-ExpiredLabel

ระบบประเมินความแม่นยำของโมเดล OCR สำหรับข้อมูลฉลากสินค้า โดยเปรียบเทียบข้อความที่โมเดลอ่านได้กับ Label
ที่ผู้ใช้กรอก (Production, Time, Expiry, Code) และแสดงอันดับความแม่นยำบนหน้าเว็บ

## วิธีใช้งาน

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

เปิดเบราว์เซอร์ที่ `http://localhost:8000`

## หมายเหตุ

- หากต้องการใช้งาน RapidOCR ให้ติดตั้ง `rapidocr-onnxruntime`
- หากต้องการใช้งาน Doctr (เฉพาะ fast_base และ linknet_resnet34) ให้ติดตั้ง `python-doctr`
- หากต้องการใช้งาน Tesseract ให้ติดตั้ง `pytesseract` และตัวโปรแกรม `tesseract-ocr` ในระบบ
