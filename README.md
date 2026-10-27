# GPay / UPI Screenshot Extractor

A small Streamlit app that reads Google Pay / UPI payment screenshots and turns
them into an editable table you can export to CSV or Excel.

## What it extracts

Date, time, amount, status, paid to, note, bank, account last 4 digits,
UPI transaction ID, receiver name, sender name, Google transaction ID.

## How it works

1. **OCR** (RapidOCR, runs fully offline on your CPU) reads the text lines.
2. **Space restoration:** the OCR engine drops spaces, so the app measures the
   blank pixel gaps in each line, cuts it into words and reads each word again.
   No dictionary or word list is needed.
3. **Parsing:** fields are found by their labels ("UPI transaction ID",
   "To:", "From:") and formats (12-digit UPI ID, date patterns), not by fixed
   positions.
4. Results appear in an editable table. Rows with missing fields show a
   "please check" warning. Export to CSV or Excel.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac / Linux
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Open http://localhost:8501, upload one or more screenshots, click **Extract**,
fix any cell if needed, then download CSV or Excel.

## Notes and limits

- Built for GPay-style screenshots. Other apps (PhonePe, Paytm) use different
  labels, so their label names need to be added to the regexes in `parse_payment`.
- OCR can confuse look-alike characters (`0`/`O`, `I`/`l`) and may misread the
  rupee symbol. Check the highlighted rows.
- Works with `rapidocr-onnxruntime` 1.2.x and 1.4.x (Python 3.13 installs 1.2.x).
- Everything runs locally. Nothing is uploaded anywhere.

## Privacy

Payment screenshots contain bank and transaction details. Do not commit them to
the repository (the `.gitignore` already excludes images and exports).