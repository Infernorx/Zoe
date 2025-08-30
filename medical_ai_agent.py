
from __future__ import annotations

import os
import io
import re
import json
import uuid
import base64
import pdfplumber
import pytesseract
import cv2 
from PIL import Image
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# IBM watsonx.ai SDK
try:
    from ibm_watsonx_ai.foundation_models import ModelInference
    from ibm_watsonx_ai import Credentials
except Exception:
    # if the SDK isn't installed in dev env, we'll fallback to an offline mock
    ModelInference = None  # type: ignore
    Credentials = None  # type: ignore

# Unstructured handles messy PDFs/office docs (tables, layout)
try:
    from unstructured.partition.auto import partition
except Exception:
    partition = None  # type: ignore

# ---------- Env & Globals ----------
load_dotenv()

WX_API_KEY = os.getenv("WX_API_KEY")
WX_PROJECT_ID = os.getenv("WX_PROJECT_ID")
WX_SPACE_ID = os.getenv("WX_SPACE_ID")
WX_BASE_URL = os.getenv("WX_BASE_URL", "https://us-south.ml.cloud.ibm.com")
WX_MODEL_ID = os.getenv("WX_MODEL_ID", "ibm-granite/granite-3.3-8b-instruct")

DEFAULT_PATIENT_PHONE = os.getenv("DEFAULT_PATIENT_PHONE", "+911234567890")
TIMEZONE_OFFSET_IST = 5.5  # Asia/Kolkata (hours)

# Basic sanity checks
if not WX_API_KEY:
    print("[WARN] WX_API_KEY is not set. LLM calls will fail. Populate .env to enable Granite.")

# ---------- Data Schemas ----------
class AIExtraction(BaseModel):
    raw_text: str
    patient_info: Dict[str, Any] = Field(default_factory=dict)
    sections: Dict[str, str] = Field(default_factory=dict)

class AISummary(BaseModel):
    doctor_summary: str
    patient_summary: str

class AIDecision(BaseModel):
    urgency: str  # OK | URGENT | CRITICAL
    reasons: List[str] = Field(default_factory=list)
    key_findings: Dict[str, Any] = Field(default_factory=dict)

class Appointment(BaseModel):
    patient_name: str
    when_local: str
    location: str = "OPD-2, Internal Medicine"
    doctor: str = "Dr. Gupta"

class AgentOutput(BaseModel):
    extraction: AIExtraction
    summary: AISummary
    decision: AIDecision
    appointment: Optional[Appointment] = None
    notifications: List[str] = Field(default_factory=list)

# ---------- Utilities ----------
def _now_ist() -> datetime:
    return datetime.utcnow() + timedelta(hours=TIMEZONE_OFFSET_IST)

def guess_file_type(filename: str) -> str:
    ext = filename.lower().split(".")[-1]
    if ext in {"png", "jpg", "jpeg", "tif", "tiff", "bmp"}:
        return "image"
    if ext in {"pdf"}:
        return "pdf"
    return "text"

# ---------- Ingestion & OCR ----------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    # Try unstructured for robust parsing (layout aware)
    if partition is not None:
        try:
            with io.BytesIO(file_bytes) as fb:
                elements = partition(file=fb)  # Auto strategy: VLM/HighRes/Fast
            chunks = [el.text for el in elements if hasattr(el, "text") and el.text]
            if chunks:
                return "\n".join(chunks)
        except Exception as e:
            print(f"[WARN] Unstructured parse failed, falling back to pdfplumber: {e}")

    # Fallback: pdfplumber (digital PDFs)
    text = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text).strip()

def extract_text_from_image(file_bytes: bytes) -> str:
    # Use OpenCV preprocessing + Tesseract OCR
    nparr = np_from_bytes(file_bytes)
    gray = cv2.cvtColor(nparr, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # Denoise lightly
    gray = cv2.medianBlur(gray, 3)
    pil_img = Image.fromarray(gray)
    return pytesseract.image_to_string(pil_img)

def np_from_bytes(b: bytes):
    import numpy as np  # local import to keep global deps tidy
    nparr = np.frombuffer(b, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

# ---------- Lightweight structure detection ----------
PATIENT_INFO_PATTERN = re.compile(
    r"(?is)Patient\s*Name\s*[:\-]?\s*(?P<name>[A-Za-z ]+)\b.*?(?:Age|\bM\/F|Sex)\s*[:\-]?\s*(?P<age>\d{1,3})",
)

SECTION_SPLITS = [
    ("findings", re.compile(r"(?is)\b(findings|impression|assessment)\b[:\-]?")),
    ("conclusion", re.compile(r"(?is)\b(conclusion|diagnosis|summary)\b[:\-]?")),
    ("recommendations", re.compile(r"(?is)\b(recommendations?|plan)\b[:\-]?")),
]

def basic_structure(text: str) -> AIExtraction:
    patient_info: Dict[str, Any] = {}
    m = PATIENT_INFO_PATTERN.search(text)
    if m:
        patient_info = {
            "name": m.group("name").strip(),
            "age": int(m.group("age")),
        }

    # Naive sectioning: split by known headers
    sections: Dict[str, str] = {}
    for key, pattern in SECTION_SPLITS:
        match = pattern.search(text)
        if match:
            start = match.end()
            # find next header start to bound the section
            next_starts = [p.search(text, start) for _, p in SECTION_SPLITS]
            next_positions = [n.start() for n in next_starts if n]
            end = min(next_positions) if next_positions else len(text)
            sections[key] = text[start:end].strip()
    return AIExtraction(raw_text=text, patient_info=patient_info, sections=sections)

# ---------- IBM Granite client ----------
def make_granite():
    if not WX_API_KEY:
        return None
    if ModelInference is None or Credentials is None:
        print("[WARN] IBM watsonx.ai SDK not available. LLM calls will be mocked.")
        return None
    creds = Credentials(api_key=WX_API_KEY, url=WX_BASE_URL)
    params = {
        "decoding_method": "greedy",
        "max_new_tokens": 800,
        "temperature": 0.0,
    }
    return ModelInference(
        model_id=WX_MODEL_ID,  # e.g., "ibm-granite/granite-3.3-8b-instruct"
        credentials=creds,
        project_id=WX_PROJECT_ID,
        space_id=WX_SPACE_ID,
        params=params,
    )

GRANITE = make_granite()

SYSTEM_PROMPT = (
    "You are a clinical documentation assistant for Indian hospitals. "
    "Extract key clinical facts, abnormal lab values (with units), and plain‑language explanations. "
    "Return concise outputs. Avoid making up values that are not present."
)

PROMPT_SUMMARIZE = (
    "Given the medical report text below, do the following:\n"
    "1) List diseases/diagnoses mentioned (if any).\n"
    "2) List abnormal lab values with normal ranges if stated.\n"
    "3) Summarize for doctor in <=80 words.\n"
    "4) Summarize for patient in simple Hindi‑friendly English in <=80 words.\n"
    "Report Text:\n\n{report_text}"
)

PROMPT_DECIDE = (
    "Based on the report text and extracted values, classify clinical urgency as one of: OK, URGENT, CRITICAL.\n"
    "Rules of thumb (India, adult):\n"
    "- Random glucose >= 300 mg/dL => CRITICAL; 200-299 => URGENT; <200 => OK unless symptoms suggest otherwise.\n"
    "- WBC > 15000/µL with fever >38.5°C => URGENT; >25000/µL or sepsis signs => CRITICAL.\n"
    "- BP >180/120 => CRITICAL; 150-179/100-119 => URGENT.\n"
    "- Any imaging red flags (e.g., PE, large pneumothorax) => CRITICAL.\n"
    "Return JSON with fields: urgency (OK|URGENT|CRITICAL), reasons[], key_findings{...}.\n"
    "Report Text:\n\n{report_text}"
)

# ---------- Rule engine (deterministic safety net) ----------
def parse_numeric_findings(text: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    # Very simple regexes; extend for your panels
    mgdl = re.search(r"(?i)(glucose|sugar)[^\d]{0,10}(\d{2,4})\s*mg\/?dL", text)
    if mgdl:
        out["glucose_mg_dL"] = float(mgdl.group(2))
    wbc = re.search(r"(?i)WBC[^\d]{0,10}(\d{3,5})\s*(?:/\u00B5?L|x?10\^?3?/\u00B5?L)?", text)
    if wbc:
        out["WBC_per_uL"] = float(wbc.group(1))
    systolic = re.search(r"(?i)(SBP|Systolic)[^\d]{0,10}(\d{2,3})", text)
    diastolic = re.search(r"(?i)(DBP|Diastolic)[^\d]{0,10}(\d{2,3})", text)
    if systolic and diastolic:
        out["SBP"] = float(systolic.group(2))
        out["DBP"] = float(diastolic.group(2))
    return out

def deterministic_triage(vals: Dict[str, float]) -> AIDecision:
    reasons = []
    urgency = "OK"
    if (g := vals.get("glucose_mg_dL")) is not None:
        if g >= 300:
            urgency = "CRITICAL"; reasons.append(f"Glucose {g} mg/dL >= 300")
        elif g >= 200:
            urgency = max(urgency, "URGENT", key=["OK","URGENT","CRITICAL"].index); reasons.append(f"Glucose {g} mg/dL between 200-299")
    if (w := vals.get("WBC_per_uL")) is not None:
        if w >= 25000:
            urgency = "CRITICAL"; reasons.append(f"WBC {w}/µL >= 25000")
        elif w > 15000:
            urgency = max(urgency, "URGENT", key=["OK","URGENT","CRITICAL"].index); reasons.append(f"WBC {w}/µL > 15000")
    if (sbp := vals.get("SBP")) and (dbp := vals.get("DBP")):
        if sbp > 180 or dbp > 120:
            urgency = "CRITICAL"; reasons.append(f"Hypertensive crisis {sbp}/{dbp}")
        elif sbp >= 150 or dbp >= 100:
            urgency = max(urgency, "URGENT", key=["OK","URGENT","CRITICAL"].index); reasons.append(f"Elevated BP {sbp}/{dbp}")
    return AIDecision(urgency=urgency, reasons=reasons, key_findings=vals)

# ---------- Appointment & Notification (mocks) ----------
def mock_book_appointment(patient_name: str, urgency: str) -> Appointment:
    now = _now_ist()
    if urgency == "CRITICAL":
        when = now + timedelta(hours=2)
        slot = when.replace(minute=0, second=0, microsecond=0)
    elif urgency == "URGENT":
        # Next‑day 10:30 AM IST
        next_day = (now + timedelta(days=1)).date()
        slot = datetime(next_day.year, next_day.month, next_day.day, 10, 30)
    else:
        # 3 days later 11:00 AM IST
        date = (now + timedelta(days=3)).date()
        slot = datetime(date.year, date.month, date.day, 11, 0)
    return Appointment(patient_name=patient_name or "Patient", when_local=slot.strftime("%Y-%m-%d %H:%M IST"))

def mock_notify(patient_phone: str, message: str) -> str:
    # In production, integrate with SMS gateway (e.g., Gupshup, Twilio) or WhatsApp BSP.
    print(f"[MOCK SMS -> {patient_phone}] {message}")
    return f"SMS to {patient_phone}: {message}"

# ---------- LLM helpers ----------
def llm_generate(prompt: str) -> str:
    if GRANITE is None:
        # Offline/development fallback
        return "[LLM OFFLINE MOCK RESPONSE] " + prompt[:200]
    resp = GRANITE.generate_text(prompt=prompt, guardrails=False, moderations=False)
    # SDK returns dict; handle both dict and str
    if isinstance(resp, dict):
        return resp.get("results", [{}])[0].get("generated_text", "")
    return str(resp)

# ---------- Core pipeline ----------
def process_report_bytes(filename: str, file_bytes: bytes, patient_name: str = "", patient_age: Optional[int] = None, patient_phone: Optional[str] = None) -> AgentOutput:
    ftype = guess_file_type(filename)
    if ftype == "pdf":
        text = extract_text_from_pdf(file_bytes)
    elif ftype == "image":
        text = extract_text_from_image(file_bytes)
    else:
        text = file_bytes.decode("utf-8", errors="ignore")

    extraction = basic_structure(text)

    # Summaries via Granite
    summ_prompt = f"System: {SYSTEM_PROMPT}\n\n" + PROMPT_SUMMARIZE.format(report_text=text[:10000])
    gen = llm_generate(summ_prompt)

    # Heuristics to split doctor/patient summaries from model output
    doctor_summary = extract_section(gen, ["Doctor", "Doctor's", "For doctor"]) or gen.strip()[:300]
    patient_summary = extract_section(gen, ["Patient", "Patient's", "For patient"]) or gen.strip()[:300]

    # Deterministic triage + LLM check
    vals = parse_numeric_findings(text)
    det = deterministic_triage(vals)

    decide_prompt = f"System: {SYSTEM_PROMPT}\n\n" + PROMPT_DECIDE.format(report_text=text[:8000])
    llm_json = safe_json(llm_generate(decide_prompt))

    # Merge decisions (worst wins)
    order = {"OK": 0, "URGENT": 1, "CRITICAL": 2}
    llm_urg = llm_json.get("urgency", "OK")
    final_urg = max([det.urgency, llm_urg], key=lambda u: order.get(u, 0))
    reasons = list(dict.fromkeys(det.reasons + llm_json.get("reasons", [])))
    key_findings = {**det.key_findings, **llm_json.get("key_findings", {})}

    decision = AIDecision(urgency=final_urg, reasons=reasons, key_findings=key_findings)

    # Book & notify
    appt = mock_book_appointment(patient_name or extraction.patient_info.get("name", ""), decision.urgency)

    phone = patient_phone or DEFAULT_PATIENT_PHONE
    notify_msgs = []
    notify_msgs.append(mock_notify(phone, f"Hello {appt.patient_name}, your report was reviewed. Urgency: {decision.urgency}."))
    notify_msgs.append(mock_notify(phone, f"Appointment booked: {appt.when_local} at {appt.location} with {appt.doctor}."))

    # Patient‑friendly summary SMS (shortened)
    brief_patient = patient_summary
    if len(brief_patient) > 200:
        brief_patient = brief_patient[:197] + "..."
    notify_msgs.append(mock_notify(phone, brief_patient))

    return AgentOutput(
        extraction=extraction,
        summary=AISummary(doctor_summary=doctor_summary, patient_summary=patient_summary),
        decision=decision,
        appointment=appt,
        notifications=notify_msgs,
    )

# ---------- Small helpers ----------
def extract_section(text: str, headers: List[str]) -> str | None:
    for h in headers:
        m = re.search(rf"(?is){re.escape(h)}[^\n]*[:\-]?\n?(.+)$", text)
        if m:
            return m.group(1).strip()
    return None

def safe_json(text: str) -> Dict[str, Any]:
    # Try to locate JSON in text
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        try:
            # Relaxed fixes for trailing commas etc.
            cleaned = re.sub(r",\s*([}\]])", r"\1", m.group(0))
            return json.loads(cleaned)
        except Exception:
            return {}

# ---------- FastAPI ----------
app = FastAPI(title="Hospital Report Agent (Granite)", version="0.1.0")

class ProcessResponse(BaseModel):
    agent_run_id: str
    output: AgentOutput

@app.post("/process_report", response_model=ProcessResponse)
async def process_report(
    file: UploadFile = File(...),
    patient_name: str = Form(""),
    patient_age: Optional[int] = Form(None),
    patient_phone: Optional[str] = Form(None),
):
    data = await file.read()
    run_id = str(uuid.uuid4())
    out = process_report_bytes(file.filename, data, patient_name, patient_age, patient_phone)
    return JSONResponse(
        content={
            "agent_run_id": run_id,
            "output": json.loads(out.model_dump_json()),
        }
    )

# ---------- Local test ----------
if __name__ == "__main__":
    # Quick smoke test with sample text
    sample_text = (
        "Patient Name: Ramesh\nAge: 52\n\nFindings: Blood sugar (random) 350 mg/dL. \n"
        "WBC 15000/µL. No chest pain.\nConclusion: Poor glycemic control.\n"
        "Recommendations: Start metformin, visit doctor in 1 day."
    ).encode()
    result = process_report_bytes("sample.txt", sample_text, patient_name="Ramesh", patient_age=52, patient_phone=DEFAULT_PATIENT_PHONE)
    print(json.dumps(json.loads(result.model_dump_json()), indent=2))
