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

ระบบประเมินจะใช้รายชื่อโมเดลการแข่งขันดังนี้
- Doctr fast_base
- Doctr linknet_resnet34
- RapidOCR
- PP-OCRv3 Mobile

### สคริปต์ติดตั้งโมเดล

รันสคริปต์ด้านล่างเพื่อติดตั้งแพ็กเกจที่จำเป็นสำหรับโมเดลทั้งหมด:

```bash
bash scripts/install_ocr_models.sh
```
