import base64
import hashlib
import inspect
import io
import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from rapidocr_onnxruntime import RapidOCR

try:                                   # PDF support: pip install pymupdf
    import pymupdf
except ImportError:
    try:
        import fitz as pymupdf         # older PyMuPDF versions
    except ImportError:
        pymupdf = None

COLUMNS = [
    "Date", "Time", "Amount (INR)", "Status", "Paid To", "Note",
    "Bank", "Account Last4", "UPI Transaction ID",
    "Receiver Name", "Sender Name",
    "Google Transaction ID",
]

# Last table column: thumbnail of the row's screenshot (double-click to enlarge).
# It is shown in the table only and is never exported.
PREVIEW = "Screenshot"

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
    ref = str(r["UPI Transaction ID"])
    is_upi = re.fullmatch(r"\d{12}", ref)                           # UPI: 12 digits
    is_utr = re.fullmatch(r"[A-Za-z]{4}[A-Za-z0-9]{8,24}", ref)      # bank transfer UTR (NEFT/IMPS/RTGS)
    if not (is_upi or is_utr):
        bad.append("UPI Transaction ID")
    if not is_utr and not r["Google Transaction ID"]:                # bank transfers have no Google ID
        bad.append("Google Transaction ID")
    return bad


# ------------------------------------------------------- duplicates / preview
def is_duplicate(row, rows):
    """True if this payment is already in the list: same UPI transaction ID,
    or exactly the same extracted data (for screenshots without a readable ID)."""
    upi = row["UPI Transaction ID"]
    if upi and any(r["UPI Transaction ID"] == upi for r in rows):
        return True
    has_data = row["Date"] != "" or row["Amount (INR)"] != ""
    return has_data and any(all(r[c] == row[c] for c in COLUMNS) for r in rows)


def make_preview(img: Image.Image, max_width=600) -> str:
    """Screenshot as a small base64 JPEG, for the table's Screenshot column."""
    im = img.convert("RGB")
    if im.width > max_width:
        im = im.resize((max_width, int(im.height * max_width / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


# # ----------------------------------------------------------------- UI
# st.set_page_config(page_title="GPay Screenshot Extractor", layout="wide")
# st.title("GPay / UPI Screenshot → Table")
# st.caption("Upload payment screenshots. Details are extracted into a table you can edit and export.")

# if "rows" not in st.session_state:
#     st.session_state.rows = []
# if "seen_files" not in st.session_state:
#     st.session_state.seen_files = set()      # fingerprints of screenshots already read

# files = st.file_uploader(
#     "Upload screenshots", type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True
# )

# col_a, col_b = st.columns([1, 1])
# if col_a.button("Extract", type="primary", disabled=not files):
#     progress = st.progress(0.0)
#     for n, f in enumerate(files, 1):
#         data = f.getvalue()
#         digest = hashlib.sha256(data).hexdigest()
#         if digest not in st.session_state.seen_files:       # the exact same file again -> skip
#             st.session_state.seen_files.add(digest)
#             img = Image.open(io.BytesIO(data))
#             row = parse_payment(ocr_lines(img), f.name)
#             if not is_duplicate(row, st.session_state.rows):
#                 row[PREVIEW] = make_preview(img)
#                 st.session_state.rows.append(row)
#         progress.progress(n / len(files))
#     progress.empty()

# if col_b.button("Clear all"):
#     st.session_state.rows = []
#     st.session_state.seen_files = set()

# # taller rows so the thumbnails are visible (only on Streamlit versions that support it)
# ROW_KW = {"row_height": 80} if "row_height" in inspect.signature(st.data_editor).parameters else {}

# if st.session_state.rows:
#     df = pd.DataFrame(st.session_state.rows, columns=COLUMNS + [PREVIEW])

#     st.subheader("Extracted data (click a cell to correct it)")
#     st.caption("Double-click a thumbnail in the Screenshot column to open the full screenshot and verify the row.")
#     edited = st.data_editor(
#         df,
#         num_rows="dynamic",
#         use_container_width=True,
#         disabled=[PREVIEW],
#         column_config={PREVIEW: st.column_config.ImageColumn(PREVIEW)},
#         **ROW_KW,
#     )
#     table = edited.drop(columns=[PREVIEW], errors="ignore")   # what gets exported

#     for i, r in table.fillna("").iterrows():
#         if any(str(v) != "" for v in r):
#             bad = check_row(r)
#             if bad:
#                 st.warning(f"Row {i + 1}: please check {', '.join(bad)}")

#     csv_bytes = table.to_csv(index=False).encode("utf-8-sig")
#     xlsx_buf = io.BytesIO()
#     with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
#         table.to_excel(writer, index=False, sheet_name="Payments")
#     d1, d2 = st.columns(2)
#     d1.download_button("Download CSV", csv_bytes, "payments.csv", "text/csv")
#     d2.download_button(
#         "Download Excel", xlsx_buf.getvalue(), "payments.xlsx",
#         "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
#     )

# ------------------------------------------------------------ PDF support
# Bank transfer advice (e.g. "Domestic Payment Transaction Details"): label -> pattern
BANK_LABELS = [
    ("utr",         r"(?:utr|transaction\s*ref(?:erence)?|reference)\s*(?:no\.?|number|id)?"),
    ("from_acct",   r"from\s*account(?:\s*(?:number|no\.?))?"),
    ("txn_date",    r"transaction\s*date(?:\s*(?:and|&)\s*time)?"),
    ("benef",       r"beneficiary\s*name|payee\s*name"),
    ("to_name",     r"to\s*account\s*name"),
    ("desc",        r"description|remarks?|narration|purpose"),
    ("status",      r"status"),
    ("amount",      r"amount"),
    ("total_debit", r"total\s*debit(?:\s*from\s*account)?"),
    ("when",        r"transfer\s*when"),
]
ACRONYMS = {"HDFC", "ICICI", "SBI", "IDFC", "PNB", "UCO", "IDBI", "RBL", "DBS", "HSBC", "BOB", "AU"}


def rows_from_items(items):
    """items = (x, y_center, height, text) pieces. Join the pieces that sit on
    the same visual line, left to right -> list of line strings."""
    groups = []
    for it in sorted(items, key=lambda t: t[1]):
        if groups and abs(it[1] - groups[-1]["y"]) <= 0.6 * max(it[2], groups[-1]["h"]):
            g = groups[-1]
            g["items"].append(it)
            g["y"] = sum(i[1] for i in g["items"]) / len(g["items"])
            g["h"] = max(g["h"], it[2])
        else:
            groups.append({"y": it[1], "h": it[2], "items": [it]})
    return [" ".join(i[3] for i in sorted(g["items"], key=lambda t: t[0])).strip() for g in groups]


def bank_fields(rows):
    """Find 'Label  value' rows of a bank advice -> {key: value}."""
    found = {}
    for r in rows:
        for key, pat in BANK_LABELS:
            if key not in found:
                m = re.match(rf"^(?:{pat})(?:\s*[:\-]\s*|\s+)(.+)$", r, re.I)
                if m:
                    found[key] = m.group(1).strip()
                    break
    return found


def find_bank_name(rows):
    """The paying bank = a line that is just a bank name (letterhead / logo).
    The 'Bank Name' row is the other side's bank, so it is skipped."""
    for r in rows:
        if re.match(r"^bank\s*name\b", r, re.I):
            continue
        m = re.fullmatch(r"([A-Za-z&.]+(?:\s+[A-Za-z&.]+){0,2}\s+bank(?:\s+(?:ltd\.?|limited))?)", r.strip(), re.I)
        if m:
            words = re.sub(r"\s+(ltd\.?|limited)$", "", m.group(1), flags=re.I).split()
            return " ".join(w.upper() if w.upper() in ACRONYMS else w.capitalize() for w in words)
    return ""


def parse_bank_advice(rows, payer_bank=""):
    """Bank transfer advice -> table row (same columns as a GPay screenshot),
    or None if these lines are not a bank advice."""
    f = bank_fields(rows)
    amt = re.search(r"\d[\d,]*(?:\.\d+)?", f.get("amount") or f.get("total_debit", ""))
    utr = clean_id(f.get("utr", ""))
    if not (amt and (utr or len(f) >= 4)):
        return None
    row = {c: "" for c in COLUMNS}
    row["Amount (INR)"] = float(amt.group(0).replace(",", ""))
    # OCR can read "01:38" as "01 :38", so tidy the colon before reading the time
    row["Date"], row["Time"] = parse_date_time(re.sub(r"(\d)\s*:\s*(\d)", r"\1:\2", f.get("txn_date", "")))
    row["Time"] = re.sub(r"^0(\d:)", r"\1", row["Time"])            # 01:38 PM -> 1:38 PM
    if not row["Date"]:                                              # fall back to "Transfer When" / the intro sentence
        for src in [f.get("when", "")] + rows:
            row["Date"], _ = parse_date_time(src)
            if row["Date"]:
                break
    row["Status"] = f.get("status", "").title()
    name = f.get("benef") or f.get("to_name", "")
    row["Paid To"] = name
    row["Receiver Name"] = name
    row["Note"] = f.get("desc", "")
    row["UPI Transaction ID"] = utr
    digits = re.sub(r"\D", "", f.get("from_acct", ""))
    row["Account Last4"] = digits[-4:] if len(digits) >= 4 else ""
    row["Bank"] = payer_bank
    return row


def bank_from_logo(img):
    """Read the paying bank from the page header (logos are pictures, not text)."""
    head = img.crop((0, 0, img.width, int(img.height * 0.14)))
    lines = ocr_lines(head)
    return find_bank_name(rows_from_items([(l["x"], l["y"], l["h"], l["spaced"]) for l in lines]))


def extract_pdf(data):
    """Read every page of a PDF -> list of (row, page_image).
    Normal PDFs: exact text from the text layer. Scanned PDFs: OCR."""
    results = []
    doc = pymupdf.open(stream=data, filetype="pdf")
    for page in doc:
        zoom = 1600 / max(page.rect.width, page.rect.height)         # about as big as a phone screenshot
        pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        row = None
        words = page.get_text("words")
        if len(words) >= 15:                                         # has a real text layer -> no OCR needed
            rows = rows_from_items([(w[0], (w[1] + w[3]) / 2, w[3] - w[1], w[4]) for w in words])
            row = parse_bank_advice(rows, find_bank_name(rows))
            if row is not None and not row["Bank"]:
                row["Bank"] = bank_from_logo(img)
        if row is None:                                              # scanned / image-only page -> OCR
            lines = ocr_lines(img)
            rows = rows_from_items([(l["x"], l["y"], l["h"], l["spaced"]) for l in lines])
            row = parse_bank_advice(rows, find_bank_name(rows))
            if row is None:                                          # e.g. a GPay screenshot saved as PDF
                row = parse_payment(lines)
                if row["Amount (INR)"] == "" and row["Date"] == "":
                    row = None
        if row is not None:
            results.append((row, img))
    return results

# ----------------------------------------------------------------- UI
st.set_page_config(page_title="GPay Screenshot Extractor", page_icon="🧾", layout="wide")

# STYLE = """
# <style>
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

# :root {
#   --bg: #0A0F1F; --surface: #10172B; --surface-2: #151D36; --line: #222B4A;
#   --text: #E8ECF8; --muted: #8E99BA; --accent: #6366F1; --accent-2: #8B5CF6;
#   --good: #34D399; --warn: #F59E0B;
# }

# /* ---------- page ---------- */
# html, body, .stApp, [data-testid="stAppViewContainer"], button, input, textarea {
#   font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif !important;
# }
# .stApp {
#   background:
#     radial-gradient(1100px 520px at 8% -8%, rgba(99,102,241,.20), transparent 60%),
#     radial-gradient(900px 480px at 100% 0%, rgba(139,92,246,.15), transparent 55%),
#     var(--bg);
#   color: var(--text);
# }
# [data-testid="stHeader"] { background: transparent; }
# [data-testid="stAppDeployButton"], #MainMenu, footer { display: none !important; }
# .block-container { padding: 2.4rem 3rem 4rem; max-width: 2400px; }
# ::selection { background: rgba(99,102,241,.45); }
# * { scrollbar-width: thin; scrollbar-color: #2B3560 transparent; }

# /* ---------- hero ---------- */
# .hero {
#   display: flex; align-items: center; gap: 18px; flex-wrap: wrap;
#   padding: 26px 30px; margin-bottom: 22px; border-radius: 20px;
#   background: linear-gradient(135deg, rgba(99,102,241,.16), rgba(139,92,246,.08) 55%, rgba(16,23,43,.6));
#   border: 1px solid rgba(99,102,241,.28);
#   box-shadow: 0 10px 40px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.04);
# }
# .hero-icon {
#   width: 54px; height: 54px; border-radius: 15px; flex: none;
#   display: grid; place-items: center;
#   background: linear-gradient(135deg, var(--accent), var(--accent-2));
#   box-shadow: 0 8px 22px rgba(99,102,241,.45);
# }
# .hero-text { flex: 1 1 380px; min-width: 260px; }
# .hero h1 {
#   margin: 0; padding: 0; font-size: 1.85rem; font-weight: 800; letter-spacing: -.02em; line-height: 1.2;
#   background: linear-gradient(90deg, #FFFFFF 10%, #C4C9FF 90%);
#   -webkit-background-clip: text; background-clip: text; color: transparent;
# }
# .hero h1 span { color: #A5A9FF; -webkit-text-fill-color: #A5A9FF; }
# .hero p { margin: .35rem 0 0; color: var(--muted); font-size: .98rem; }
# .chips { display: flex; gap: 8px; flex-wrap: wrap; }
# .chips span {
#   font-size: .74rem; font-weight: 600; letter-spacing: .02em; color: #C9CEFF;
#   padding: 6px 12px; border-radius: 999px;
#   background: rgba(99,102,241,.14); border: 1px solid rgba(99,102,241,.32);
# }

# /* ---------- cards ---------- */
# .st-key-card_upload, .st-key-card_table, .st-key-card_export, .empty {
#   background: linear-gradient(180deg, rgba(21,29,54,.82), rgba(16,23,43,.82));
#   border: 1px solid var(--line); border-radius: 18px;
#   padding: 22px 26px 24px; margin-bottom: 18px;
#   box-shadow: 0 12px 34px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.035);
# }

# /* ---------- step headers ---------- */
# .step { display: flex; align-items: flex-start; gap: 14px; margin: 0 0 14px; }
# .step .num {
#   flex: none; width: 30px; height: 30px; border-radius: 10px; display: grid; place-items: center;
#   font-weight: 700; font-size: .9rem; color: #fff;
#   background: linear-gradient(135deg, var(--accent), var(--accent-2));
#   box-shadow: 0 4px 12px rgba(99,102,241,.4);
# }
# .step .ttl { font-size: 1.12rem; font-weight: 700; letter-spacing: -.01em; color: var(--text); line-height: 1.35; }
# .step .hint { margin-top: 2px; font-size: .86rem; color: var(--muted); }

# /* ---------- file uploader ---------- */
# [data-testid="stFileUploaderDropzone"] {
#   background: rgba(99,102,241,.06); border: 1.5px dashed #3A468A; border-radius: 14px; padding: 18px 20px;
#   transition: border-color .15s, background .15s;
# }
# [data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent); background: rgba(99,102,241,.11); }
# [data-testid="stFileUploaderDropzoneInstructions"] span { color: #C9CEE8; font-weight: 500; }
# [data-testid="stFileUploaderDropzoneInstructions"] small { color: var(--muted); }
# [data-testid="stFileUploaderFile"] {
#   background: rgba(255,255,255,.035); border: 1px solid var(--line); border-radius: 10px; padding: 4px 8px;
# }

# /* ---------- buttons ---------- */
# [data-testid="stBaseButton-primary"] {
#   background: linear-gradient(135deg, var(--accent), var(--accent-2)); color: #fff; border: 0;
#   border-radius: 11px; padding: .58rem 1.5rem; font-weight: 600; letter-spacing: .01em;
#   box-shadow: 0 8px 20px rgba(99,102,241,.38); transition: transform .12s, box-shadow .12s, filter .12s;
# }
# [data-testid="stBaseButton-primary"]:hover { transform: translateY(-1px); filter: brightness(1.1); box-shadow: 0 12px 26px rgba(99,102,241,.5); }
# [data-testid="stBaseButton-primary"]:active { transform: translateY(0); }
# [data-testid="stBaseButton-primary"]:disabled { opacity: .4; box-shadow: none; filter: grayscale(.35); transform: none; }
# [data-testid="stBaseButton-secondary"] {
#   background: rgba(255,255,255,.03); color: #CBD1EE; border: 1px solid #2E3A66;
#   border-radius: 11px; padding: .58rem 1.3rem; font-weight: 500; transition: all .12s;
# }
# [data-testid="stBaseButton-secondary"]:hover { border-color: var(--accent); color: #fff; background: rgba(99,102,241,.14); }
# .st-key-card_export [data-testid="stBaseButton-secondary"] {
#   background: rgba(99,102,241,.12); border-color: rgba(99,102,241,.45); color: #E3E6FF; font-weight: 600;
# }
# .st-key-card_export [data-testid="stBaseButton-secondary"]:hover { background: rgba(99,102,241,.26); }

# /* ---------- table ---------- */
# [data-testid="stDataFrame"], [data-testid="stDataEditor"] {
#   border: 1px solid var(--line); border-radius: 14px; overflow: hidden;
#   box-shadow: 0 6px 22px rgba(0,0,0,.3);
# }

# /* ---------- alerts (row warnings) ---------- */
# [data-testid="stAlert"] { border-radius: 12px; margin-top: 4px; }
# [data-testid="stAlertContainer"] {
#   background: rgba(245,158,11,.09) !important; border: 1px solid rgba(245,158,11,.32);
#   border-left: 4px solid var(--warn); border-radius: 12px; color: #FCE7B2;
# }

# /* ---------- empty state ---------- */
# .empty { text-align: center; padding: 44px 24px; color: var(--muted); }
# .empty .big { font-size: 1.1rem; font-weight: 700; color: var(--text); margin-bottom: 4px; }
# .empty .ico { font-size: 2rem; margin-bottom: 6px; opacity: .9; }
# </style>
# """

STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  --bg: #140E12; --surface: #1C1319; --surface-2: #24181F; --line: #3A2A33;
  --text: #F3ECF0; --muted: #B09AA6; --accent: #7E5E71; --accent-2: #6F5061;
  --good: #C79BB3; --warn: #F59E0B;
}

/* ---------- page ---------- */
html, body, .stApp, [data-testid="stAppViewContainer"], button, input, textarea {
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif !important;
}
.stApp {
  background:
    radial-gradient(1100px 520px at 8% -8%, rgba(150,108,130,.20), transparent 60%),
    radial-gradient(900px 480px at 100% 0%, rgba(111,80,97,.15), transparent 55%),
    var(--bg);
  color: var(--text);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppDeployButton"], #MainMenu, footer { display: none !important; }
.block-container { padding: 2.4rem 3rem 4rem; max-width: 2400px; }
::selection { background: rgba(150,108,130,.45); }
* { scrollbar-width: thin; scrollbar-color: #4B3643 transparent; }

/* ---------- hero ---------- */
.hero {
  display: flex; align-items: center; gap: 18px; flex-wrap: wrap;
  padding: 26px 30px; margin-bottom: 22px; border-radius: 20px;
  background: linear-gradient(135deg, rgba(150,108,130,.16), rgba(111,80,97,.08) 55%, rgba(28,19,25,.6));
  border: 1px solid rgba(150,108,130,.28);
  box-shadow: 0 10px 40px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.04);
}
.hero-icon {
  width: 54px; height: 54px; border-radius: 15px; flex: none;
  display: grid; place-items: center;
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  box-shadow: 0 8px 22px rgba(150,108,130,.45);
}
.hero-text { flex: 1 1 380px; min-width: 260px; }
.hero h1 {
  margin: 0; padding: 0; font-size: 1.85rem; font-weight: 800; letter-spacing: -.02em; line-height: 1.2;
  background: linear-gradient(90deg, #FFFFFF 10%, #E6D3DE 90%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero h1 span { color: #D9B6C9; -webkit-text-fill-color: #D9B6C9; }
.hero p { margin: .35rem 0 0; color: var(--muted); font-size: .98rem; }
.chips { display: flex; gap: 8px; flex-wrap: wrap; }
.chips span {
  font-size: .74rem; font-weight: 600; letter-spacing: .02em; color: #E6D3DE;
  padding: 6px 12px; border-radius: 999px;
  background: rgba(150,108,130,.14); border: 1px solid rgba(150,108,130,.32);
}

/* ---------- cards ---------- */
.st-key-card_upload, .st-key-card_table, .st-key-card_export, .empty {
  background: linear-gradient(180deg, rgba(150,108,130,.82), rgba(28,19,25,.82));
  border: 1px solid var(--line); border-radius: 18px;
  padding: 22px 26px 24px; margin-bottom: 18px;
  box-shadow: 0 12px 34px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.035);
}

/* ---------- step headers ---------- */
.step { display: flex; align-items: flex-start; gap: 14px; margin: 0 0 14px; }
.step .num {
  flex: none; width: 30px; height: 30px; border-radius: 10px; display: grid; place-items: center;
  font-weight: 700; font-size: .9rem; color: #fff;
  background: linear-gradient(135deg, var(--accent), var(--accent-2));
  box-shadow: 0 4px 12px rgba(150,108,130,.4);
}
.step .ttl { font-size: 1.12rem; font-weight: 700; letter-spacing: -.01em; color: var(--text); line-height: 1.35; }
.step .hint { margin-top: 2px; font-size: .86rem; color: var(--muted); }

/* ---------- file uploader ---------- */
[data-testid="stFileUploaderDropzone"] {
  background: rgba(150,108,130,.06); border: 1.5px dashed #6F5061; border-radius: 14px; padding: 18px 20px;
  transition: border-color .15s, background .15s;
}
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent); background: rgba(150,108,130,.11); }
[data-testid="stFileUploaderDropzoneInstructions"] span { color: #E8D8E1; font-weight: 500; }
[data-testid="stFileUploaderDropzoneInstructions"] small { color: var(--muted); }
[data-testid="stFileUploaderFile"] {
  background: rgba(255,255,255,.035); border: 1px solid var(--line); border-radius: 10px; padding: 4px 8px;
}

/* ---------- buttons ---------- */
[data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, var(--accent), var(--accent-2)); color: #fff; border: 0;
  border-radius: 11px; padding: .58rem 1.5rem; font-weight: 600; letter-spacing: .01em;
  box-shadow: 0 8px 20px rgba(150,108,130,.38); transition: transform .12s, box-shadow .12s, filter .12s;
}
[data-testid="stBaseButton-primary"]:hover { transform: translateY(-1px); filter: brightness(1.1); box-shadow: 0 12px 26px rgba(150,108,130,.5); }
[data-testid="stBaseButton-primary"]:active { transform: translateY(0); }
[data-testid="stBaseButton-primary"]:disabled { opacity: .4; box-shadow: none; filter: grayscale(.35); transform: none; }
[data-testid="stBaseButton-secondary"] {
  background: rgba(255,255,255,.03); color: #CFE6DC; border: 1px solid #4B3643;
  border-radius: 11px; padding: .58rem 1.3rem; font-weight: 500; transition: all .12s;
}
[data-testid="stBaseButton-secondary"]:hover { border-color: var(--accent); color: #fff; background: rgba(150,108,130,.14); }
.st-key-card_export [data-testid="stBaseButton-secondary"] {
  background: rgba(150,108,130,.12); border-color: rgba(150,108,130,.45); color: #F6E9F0; font-weight: 600;
}
.st-key-card_export [data-testid="stBaseButton-secondary"]:hover { background: rgba(150,108,130,.26); }

/* ---------- table ---------- */
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {
  border: 1px solid var(--line); border-radius: 14px; overflow: hidden;
  box-shadow: 0 6px 22px rgba(0,0,0,.3);
}

/* ---------- alerts (row warnings) ---------- */
[data-testid="stAlert"] { border-radius: 12px; margin-top: 4px; }
[data-testid="stAlertContainer"] {
  background: rgba(245,158,11,.09) !important; border: 1px solid rgba(245,158,11,.32);
  border-left: 4px solid var(--warn); border-radius: 12px; color: #FCE7B2;
}

/* ---------- empty state ---------- */
.empty { text-align: center; padding: 44px 24px; color: var(--muted); }
.empty .big { font-size: 1.1rem; font-weight: 700; color: var(--text); margin-bottom: 4px; }
.empty .ico { font-size: 2rem; margin-bottom: 6px; opacity: .9; }
</style>
"""
st.markdown(STYLE, unsafe_allow_html=True)

HERO = (
    '<div class="hero">'
    '<div class="hero-icon"><svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="white" '
    'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M6 2h12v20l-3-2-3 2-3-2-3 2z"/><path d="M9 8h6M9 12h6"/></svg></div>'
    '<div class="hero-text"><h1>GPay / UPI Screenshot <span>→</span> Table</h1>'
    '<p>Upload payment screenshots. Details are extracted into a table you can edit and export.</p></div>'
    '<div class="chips"><span>Runs locally</span><span>CSV &amp; Excel export</span></div>'
    '</div>'
)
st.markdown(HERO, unsafe_allow_html=True)


def step(num, title, hint=""):
    """Numbered section header (looks only)."""
    html = f'<div class="step"><span class="num">{num}</span><div><div class="ttl">{title}</div>'
    if hint:
        html += f'<div class="hint">{hint}</div>'
    st.markdown(html + "</div></div>", unsafe_allow_html=True)


if "rows" not in st.session_state:
    st.session_state.rows = []
if "seen_files" not in st.session_state:
    st.session_state.seen_files = set()      # fingerprints of screenshots already read

with st.container(key="card_upload"):
    step(1, "Upload screenshots or PDFs", "PNG, JPG, WebP or PDF · you can select several files at once")
    files = st.file_uploader(
        "Upload screenshots or PDFs", type=["png", "jpg", "jpeg", "webp", "pdf"], accept_multiple_files=True,
        label_visibility="collapsed",
    )

    col_a, col_b, _ = st.columns([1.2, 1.2, 6])
    if col_a.button("Extract", type="primary", disabled=not files, use_container_width=True):
        progress = st.progress(0.0)
        for n, f in enumerate(files, 1):
            data = f.getvalue()
            digest = hashlib.sha256(data).hexdigest()
            if digest not in st.session_state.seen_files:       # the exact same file again -> skip
                is_pdf = f.name.lower().endswith(".pdf")
                results = None
                if is_pdf and pymupdf is None:
                    st.error("PDF support needs PyMuPDF. Run:  pip install pymupdf")
                elif is_pdf:
                    try:
                        results = extract_pdf(data)
                        if not results:
                            st.warning(f"{f.name}: no payment details found in this PDF.")
                    except Exception as e:
                        st.error(f"{f.name}: could not read this PDF ({e})")
                else:
                    img = Image.open(io.BytesIO(data))
                    results = [(parse_payment(ocr_lines(img), f.name), img)]
                if results is not None:
                    st.session_state.seen_files.add(digest)
                    for row, pic in results:
                        if not is_duplicate(row, st.session_state.rows):
                            row[PREVIEW] = make_preview(pic, max_width=1100 if is_pdf else 600)
                            st.session_state.rows.append(row)
            progress.progress(n / len(files))
        progress.empty()

    if col_b.button("Clear all", use_container_width=True):
        st.session_state.rows = []
        st.session_state.seen_files = set()

# taller rows so the thumbnails are visible (only on Streamlit versions that support it)
ROW_KW = {"row_height": 80} if "row_height" in inspect.signature(st.data_editor).parameters else {}

if st.session_state.rows:
    with st.container(key="card_table"):
        step(
            2, "Extracted data (click a cell to correct it)",
            "Double-click a thumbnail in the Screenshot column to open the full screenshot and verify the row.",
        )
        df = pd.DataFrame(st.session_state.rows, columns=COLUMNS + [PREVIEW])
        edited = st.data_editor(
            df,
            num_rows="dynamic",
            use_container_width=True,
            disabled=[PREVIEW],
            column_config={PREVIEW: st.column_config.ImageColumn(PREVIEW)},
            **ROW_KW,
        )
        table = edited.drop(columns=[PREVIEW], errors="ignore")   # what gets exported

        for i, r in table.fillna("").iterrows():
            if any(str(v) != "" for v in r):
                bad = check_row(r)
                if bad:
                    st.warning(f"Row {i + 1}: please check {', '.join(bad)}")

    with st.container(key="card_export"):
        step(3, "Export", "Download the table as it appears above (the Screenshot column is not included).")
        csv_bytes = table.to_csv(index=False).encode("utf-8-sig")
        xlsx_buf = io.BytesIO()
        with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
            table.to_excel(writer, index=False, sheet_name="Payments")
        d1, d2, _ = st.columns([1.2, 1.2, 6])
        d1.download_button("Download CSV", csv_bytes, "payments.csv", "text/csv", use_container_width=True)
        d2.download_button(
            "Download Excel", xlsx_buf.getvalue(), "payments.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
else:
    st.markdown(
        '<div class="empty"><div class="ico">🧾</div>'
        '<div class="big">No payments yet</div>'
        'Upload one or more screenshots above, then click <b>Extract</b>.</div>',
        unsafe_allow_html=True,
    )