import io
import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

COLUMNS = [
    "Date", "Time", "Amount (INR)", "Status", "Paid To", "Note",
    "Bank", "Account Last4", "UPI Transaction ID",
    "Receiver Name", "Sender Name",
    "Google Transaction ID",
]

# Word-space detection. Gaps are measured as a fraction of the text-line height.
# A gap can only be a space if it is at least MIN_WORD_GAP, and the spaces must
# stand clearly apart from the letter gaps in the same line (MIN_JUMP times bigger).
MIN_WORD_GAP = 0.15
MIN_JUMP = 1.5


@st.cache_resource
def get_ocr():
    return RapidOCR()


# ---------------------------------------------------------------- OCR
def find_gaps(crop):
    """Blank pixel columns inside one text-line crop -> (ink_start, ink_end, gaps)."""
    gray = crop.mean(axis=2)
    ink = (np.abs(gray - np.median(gray)) > 45).any(axis=0)
    xs = np.where(ink)[0]
    if len(xs) == 0:
        return 0, 0, []
    a, b = int(xs[0]), int(xs[-1])
    gaps, start = [], None
    for x in range(a, b + 1):
        if not ink[x]:
            if start is None:
                start = x
        elif start is not None:
            gaps.append((start, x - start))
            start = None
    return a, b, gaps


def word_gaps(gaps, h):
    """Pick the gaps that are real spaces: the widest group of gaps that is
    clearly separated from the narrower letter gaps of the same line."""
    ws = sorted((w / h for _, w in gaps), reverse=True)
    best_k, best_ratio = 0, 0.0
    for k in range(1, len(ws) + 1):
        if ws[k - 1] < MIN_WORD_GAP:
            break
        nxt = ws[k] if k < len(ws) else 0.0
        ratio = ws[k - 1] / max(nxt, 0.03)
        if ratio > best_ratio:
            best_k, best_ratio = k, ratio
    if best_ratio < MIN_JUMP:
        return []
    limit = ws[best_k - 1] - 1e-9
    return [(s, w) for s, w in gaps if w / h >= limit]


def first_text(res):
    """Dig the first plain string out of an OCR result, whatever list nesting
    this version of rapidocr uses."""
    while isinstance(res, (list, tuple)) and len(res) > 0:
        res = res[0]
    return res.strip() if isinstance(res, str) else ""


def read_words(segs):
    """Read a list of small word images with the OCR's text recognizer.
    The recognizer is called text_rec in new rapidocr versions and
    text_recognizer in older ones. Returns a list of strings (or None)."""
    ocr = get_ocr()
    for name in ("text_rec", "text_recognizer"):
        rec = getattr(ocr, name, None)
        if rec is None:
            continue
        try:
            out = rec(segs)
            items = out[0] if isinstance(out, (list, tuple)) and len(out) == 2 else out
            words = [first_text(it) for it in items]
            if len(words) == len(segs):
                return words
        except Exception:
            pass
    return None


def respace(crop, h):
    """Cut a text line into words at real pixel gaps and read each word again.
    Returns the correctly spaced text, or None if it could not be done."""
    a, b, gaps = find_gaps(crop)
    big = word_gaps(gaps, h)
    if not big:
        return None
    edges = [a] + [x for s, w in big for x in (s, s + w)] + [b + 1]
    bg = int(np.median(crop))
    segs = []
    for i in range(0, len(edges), 2):
        seg = crop[:, edges[i]:edges[i + 1]]
        segs.append(np.pad(seg, ((8, 8), (12, 12), (0, 0)), constant_values=bg))
    words = read_words(segs)
    if not words or not all(words):
        return None
    return " ".join(words)


def ocr_lines(img: Image.Image):
    """Run OCR, return text lines top-to-bottom.
    'text' = raw OCR text (best for IDs/numbers), 'spaced' = same line with
    word spaces restored from the image (best for names, notes, labels)."""
    img = img.convert("RGB")
    img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
    arr = np.array(img)
    if arr.mean() < 127:          # dark mode screenshot -> invert for better OCR
        arr = 255 - arr

    result, _ = get_ocr()(arr)
    lines = []
    for box, text, _score in result or []:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        h = max(ys) - min(ys)
        spaced = text.strip()
        if sum(c.isalpha() for c in spaced) >= 3:
            crop = arr[max(int(min(ys)) - 2, 0):int(max(ys)) + 2,
                       max(int(min(xs)) - 4, 0):int(max(xs)) + 4]
            spaced = respace(crop, h) or spaced
        lines.append({
            "text": text.strip(),
            "spaced": spaced,
            "y": sum(ys) / len(ys),
            "x": min(xs),
            "h": h,
        })
    lines.sort(key=lambda l: (l["y"], l["x"]))
    return lines


# ------------------------------------------------------------ parsing
def clean_id(s: str) -> str:
    return re.sub(r"\s+", "", s)


DATE_RE = r"(\d{1,2})\s*([A-Za-z0]{3,9})\.?\s*,?\s*(\d{4})"        # 24 Oct 2026
DATE_RE2 = r"([A-Za-z0]{3,9})\.?\s*(\d{1,2})\s*,?\s*(\d{4})"       # Oct 24, 2026
TIME_RE = r"(\d{1,2}:\d{2})\s*([AaPp][Mm])?"
STATUS_RE = r"completed|successful|success|failed|pending|processing|declined|cancelled"


def month_to_date(day, mon, year):
    mon = mon[:3].replace("0", "O").title()                  # '0ct' -> 'Oct'
    try:
        return datetime.strptime(f"{day} {mon} {year}", "%d %b %Y").strftime("%Y-%m-%d")
    except ValueError:
        return ""


def parse_date_time(text: str):
    date_out, time_out = "", ""
    d = re.search(DATE_RE, text)
    if d:
        date_out = month_to_date(d.group(1), d.group(2), d.group(3))
    if not date_out:
        d = re.search(DATE_RE2, text)
        if d:
            date_out = month_to_date(d.group(2), d.group(1), d.group(3))
    t = re.search(TIME_RE, text)
    if t:
        time_out = t.group(1) + (f" {t.group(2).upper()}" if t.group(2) else "")
    return date_out, time_out


def is_date_line(t: str) -> bool:
    return bool(re.search(TIME_RE, t) and (re.search(DATE_RE, t) or re.search(DATE_RE2, t)))


def value_after(stexts, rawtexts, start, label_regex, raw=False):
    """Find a label line at/after `start`; return (value, index_of_value_line).
    raw=True takes a value from the next line's raw OCR text (for IDs)."""
    for i in range(start, len(stexts)):
        m = re.search(label_regex, stexts[i], re.I)
        if m:
            rest = stexts[i][m.end():].strip(" :")
            if rest:
                return rest, i
            if i + 1 < len(stexts):
                return (rawtexts if raw else stexts)[i + 1], i + 1
    return "", -1


def parse_payment(lines, filename=""):
    texts = [l["text"] for l in lines]        # raw OCR  (IDs, numbers)
    st_ = [l["spaced"] for l in lines]        # re-spaced (names, labels, notes)
    row = {c: "" for c in COLUMNS}

    # ---- Amount: the biggest "numbers only" text on the screen
    amt_idx = -1
    best_h = 0
    for i, l in enumerate(lines):
        t = l["text"]
        digits = re.sub(r"[^\d.,]", "", t)
        non_digits = len(re.sub(r"[\d.,\s]", "", t))
        if digits and non_digits <= 2 and l["h"] > best_h:
            best_h, amt_idx = l["h"], i
    if amt_idx >= 0:
        raw = re.sub(r"[^\d.,]", "", texts[amt_idx]).replace(",", "")
        try:
            row["Amount (INR)"] = float(raw)
        except ValueError:
            row["Amount (INR)"] = raw

    # ---- Paid To (header line above the amount: "To <name>")
    if amt_idx > 0:
        for j in range(amt_idx - 1, -1, -1):
            if re.match(r"^to", st_[j], re.I):
                row["Paid To"] = re.sub(r"^to\s*:?\s*", "", st_[j], flags=re.I).strip()
                break

    # ---- Note: lines under the amount, up to the first of
    #      "Pay again" / status line / date line
    if amt_idx >= 0:
        end_idx = next(
            (i for i in range(amt_idx + 1, len(st_))
             if re.search(r"pay\s*again|" + STATUS_RE, st_[i], re.I) or is_date_line(st_[i])),
            -1,
        )
        if end_idx > amt_idx:
            row["Note"] = " ".join(st_[amt_idx + 1:end_idx]).strip()

    # ---- Status
    for t in st_:
        m = re.search(STATUS_RE, t, re.I)
        if m:
            row["Status"] = m.group(0).title()
            break

    # ---- Date & time
    date_idx = -1
    for i, t in enumerate(st_):
        if is_date_line(t):
            row["Date"], row["Time"] = parse_date_time(t)
            date_idx = i
            break
    start = max(date_idx, 0)

    # ---- Where the details card starts ("UPI transaction ID" label)
    upi_label_idx = next(
        (i for i in range(start, len(st_)) if re.search(r"upi\s*transaction", st_[i], re.I)),
        -1,
    )
    det_start = upi_label_idx if upi_label_idx >= 0 else start

    # ---- Bank + last 4 digits (line like "Yes Bank 3608"); prefer a line with "bank"
    cands = []
    for i in range(start + 1, upi_label_idx if upi_label_idx >= 0 else len(st_)):
        m = re.match(r"^(.*?[A-Za-z].*?)\s*(\d{4})$", st_[i])
        if m:
            cands.append((m.group(1).strip(), m.group(2)))
    cands.sort(key=lambda c: "bank" not in c[0].lower())    # stable: bank lines first
    if cands:
        row["Bank"], row["Account Last4"] = cands[0]

    # ---- UPI transaction ID (12 digits)
    v, _ = value_after(st_, texts, start, r"upi\s*transaction\s*id", raw=True)
    upi = clean_id(v)
    if not re.fullmatch(r"\d{12}", upi):      # label misread -> look for the 12-digit ID itself
        upi = next((clean_id(t) for t in texts[start:] if re.fullmatch(r"\d{12}", clean_id(t))), upi)
    row["UPI Transaction ID"] = upi

    # ---- Receiver / Sender (prefer "To:" / "From:" with colon, in the details card)
    for label, col in ((r"to", "Receiver Name"), (r"from", "Sender Name")):
        v, _ = value_after(st_, texts, det_start, rf"^{label}\s*:")
        if not v:
            v, _ = value_after(st_, texts, det_start, rf"^{label}\b\s*:?")
        row[col] = v

    # ---- Google transaction ID
    v, _ = value_after(st_, texts, start, r"google\s*transaction\s*id", raw=True)
    row["Google Transaction ID"] = clean_id(v)

    return row


def check_row(r):
    """Fields that look missing/wrong, so you know which rows to double-check."""
    bad = []
    if not r["Date"]:
        bad.append("Date")
    if not r["Time"]:
        bad.append("Time")
    if r["Amount (INR)"] in ("", None):
        bad.append("Amount")
    if not re.fullmatch(r"\d{12}", str(r["UPI Transaction ID"])):
        bad.append("UPI Transaction ID")
    if not r["Google Transaction ID"]:
        bad.append("Google Transaction ID")
    return bad


# ----------------------------------------------------------------- UI
st.set_page_config(page_title="GPay Screenshot Extractor", layout="wide")
st.title("GPay / UPI Screenshot → Table")
st.caption("Upload payment screenshots. Details are extracted into a table you can edit and export.")

if "rows" not in st.session_state:
    st.session_state.rows = []

files = st.file_uploader(
    "Upload screenshots", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True
)

col_a, col_b = st.columns([1, 1])
if col_a.button("Extract", type="primary", disabled=not files):
    progress = st.progress(0.0)
    for n, f in enumerate(files, 1):
        img = Image.open(f)
        lines = ocr_lines(img)
        st.session_state.rows.append(parse_payment(lines, f.name))
        progress.progress(n / len(files))
    progress.empty()

if col_b.button("Clear all"):
    st.session_state.rows = []

if st.session_state.rows:
    df = pd.DataFrame(st.session_state.rows, columns=COLUMNS)
    # remove duplicates (same UPI ID uploaded twice)
    has_id = df["UPI Transaction ID"] != ""
    df = pd.concat([df[has_id].drop_duplicates("UPI Transaction ID"), df[~has_id]]).reset_index(drop=True)

    st.subheader("Extracted data (click a cell to correct it)")
    edited = st.data_editor(df, num_rows="dynamic", use_container_width=True)

    for i, r in edited.fillna("").iterrows():
        if any(str(v) != "" for v in r):
            bad = check_row(r)
            if bad:
                st.warning(f"Row {i + 1}: please check {', '.join(bad)}")

    csv_bytes = edited.to_csv(index=False).encode("utf-8-sig")
    xlsx_buf = io.BytesIO()
    with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
        edited.to_excel(writer, index=False, sheet_name="Payments")
    d1, d2 = st.columns(2)
    d1.download_button("Download CSV", csv_bytes, "payments.csv", "text/csv")
    d2.download_button(
        "Download Excel", xlsx_buf.getvalue(), "payments.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )