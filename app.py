from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path
from PIL import Image
from openai import OpenAI
import streamlit as st

# NVIDIA GB10 VLM Inference Server Configuration
BASE_URL = "http://10.246.100.230/v1"
MODEL = "qwen3-vl-8b-instruct"
API_KEY = "usaadatathon"

PROJECT_ROOT = Path.cwd()
SERVER_DATASET_DIR = Path("/tmp/datathon/UTSA-flood-images-complete/images")
REPORT_JSON_PATH = PROJECT_ROOT / "triage_report.json"
REPORT_BACKUP_PATH = PROJECT_ROOT / "triage_report.backup.json"
REPORT_CSV_PATH = PROJECT_ROOT / "triage_report.csv"

LAND_USE_PROFILES = {
    "Dense Urban / Commercial": {"pop_density": "Very High (~5,500/sq mi)", "vuln_multiplier": 1.45},
    "Suburban Residential": {"pop_density": "High (~3,400/sq mi)", "vuln_multiplier": 1.30},
    "Highway / Transit Corridor": {"pop_density": "Medium-High (Transit Artery)", "vuln_multiplier": 1.25},
    "Rural / Agricultural": {"pop_density": "Low-Medium (~1,200/sq mi)", "vuln_multiplier": 1.15},
}
DEFAULT_PROFILE = {"pop_density": "Medium (~2,800/sq mi)", "vuln_multiplier": 1.20}

TRIAGE_PROMPT = """Analyze this aerial image for flood severity and surrounding land use.
Distinguish normal water bodies (rivers, bayous, ponds within natural banks = SAFE) from floodwater (brown/muddy water covering roads, yards, vehicles, or buildings).

Classify severity into EXACTLY ONE category:
1. "VERY DANGEROUS": Severe flooding. Deep muddy water submerging houses, roofs, vehicles, or completely cutting off neighborhoods.
2. "DANGEROUS": Moderate flooding. Standing water on roads, flooded yards/fields, or edges of buildings, making travel hazardous.
3. "SAFE": No visible flooding. Dry roads, dry land, or normal water bodies within banks.

Respond ONLY with valid JSON:
{
  "status": "VERY DANGEROUS" | "DANGEROUS" | "SAFE",
  "base_severity": <integer from 0 to 70>,
  "confidence": "High" | "Medium" | "Low",
  "land_use": "Dense Urban / Commercial" | "Suburban Residential" | "Highway / Transit Corridor" | "Rural / Agricultural",
  "water_type": "Muddy Floodwater" | "Standing Road Water" | "Normal River/Pond" | "None",
  "visible_infrastructure": "Short phrase listing visible roads, houses, buildings, vehicles, or bridges",
  "hazards": ["1-3 specific visible hazards"],
  "evidence": "1-2 concise sentences describing visible water, roads, vehicles, or structures.",
  "dispatch_units": ["1-3 concrete response units"],
  "citizen_alert_en": "Urgent SMS alert for residents in this sector (max 20 words).",
  "citizen_alert_es": "Spanish translation of the urgent SMS alert (max 20 words)."
}"""

COMPARE_PROMPT = """Compare Image 1 (BEFORE flood) and Image 2 (AFTER flood) of the same area.
1. Identify which specific roads, houses, or neighborhoods were dry in Image 1 and are now submerged in Image 2.
2. Estimate evacuation urgency (Critical / Moderate / Low).
Be concise, factual, and state any uncertainty."""


def image_to_data_url(image_source, max_size: int = 896) -> str:
    with Image.open(image_source) as img:
        img = img.convert("RGB")
        img.thumbnail((max_size, max_size))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=82)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def analyze_single_image(image_source, filename: str) -> dict:
    data_url = image_to_data_url(image_source)
    with OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=90, max_retries=1) as client:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            top_p=0.8,
            max_tokens=380,
            extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
            messages=[
                {"role": "system", "content": "You are an emergency disaster-response AI. Output valid JSON only."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": TRIAGE_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        )
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.choices[0].message.content.strip(), flags=re.IGNORECASE)
    res = json.loads(raw)
    land_use = res.get("land_use", "Suburban Residential")
    profile = LAND_USE_PROFILES.get(land_use, DEFAULT_PROFILE)
    base = int(res.get("base_severity", 50))
    res["file"] = filename
    res["district"] = f"Sector {Path(filename).stem.upper()} ({land_use})"
    res["pop_density"] = profile["pop_density"]
    res["critical_infra"] = res.get("visible_infrastructure", "Local Roads & Structures")
    res["priority_score"] = min(100, round(base * profile["vuln_multiplier"]))
    res["needs_human_review"] = res.get("confidence") == "Low"
    return res


def ask_vlm(image_sources, question: str) -> str:
    if not isinstance(image_sources, list):
        image_sources = [image_sources]
    content = [{"type": "text", "text": question}]
    for src in image_sources:
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(src)}})
    with OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=90, max_retries=1) as client:
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            top_p=0.8,
            max_tokens=320,
            extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
            messages=[
                {"role": "system", "content": "Use visible image evidence. Be concise and state uncertainty."},
                {"role": "user", "content": content},
            ],
        )
    return response.choices[0].message.content


def load_reports() -> list[dict]:
    for path in (REPORT_JSON_PATH, REPORT_BACKUP_PATH):
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                return sorted(data, key=lambda x: x.get("priority_score", 0), reverse=True)
    return []


st.set_page_config(page_title="FloodPulse AI — Disaster Triage", page_icon="🚨", layout="wide")

# --- SIDEBAR ---
st.sidebar.title("🚨 FloodPulse AI")
st.sidebar.caption("Qwen3-VL on NVIDIA GB10")

if st.sidebar.button("🔄 Reload Reports from Disk", use_container_width=True):
    st.session_state["results"] = load_reports()

if "results" not in st.session_state:
    st.session_state["results"] = load_reports()

results = st.session_state.get("results", [])

if results:
    st.sidebar.divider()
    st.sidebar.subheader("📥 EOC Incident Export")
    st.sidebar.download_button(
        label="Download triage_report.json",
        data=json.dumps(results, indent=2, ensure_ascii=False),
        file_name="triage_report.json",
        mime="application/json",
        use_container_width=True,
    )
    if REPORT_CSV_PATH.exists():
        st.sidebar.download_button(
            label="Download triage_report.csv",
            data=REPORT_CSV_PATH.read_text(encoding="utf-8"),
            file_name="triage_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

# --- KPI METRICS ---
st.title("🛰️ FloodPulse AI — Autonomous Aerial Disaster Triage")
st.caption("Multimodal Flood Severity Classification + GIS Land-Use Vulnerability Scoring (NVIDIA GB10)")

very_dangerous = [r for r in results if r.get("status") == "VERY DANGEROUS"]
dangerous = [r for r in results if r.get("status") == "DANGEROUS"]
safe = [r for r in results if r.get("status") == "SAFE"]
hitl_count = sum(1 for r in results if r.get("needs_human_review"))

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Sectors", len(results))
c2.metric("🔴 VERY DANGEROUS", len(very_dangerous), delta="Airboats / Helicopters", delta_color="inverse")
c3.metric("🟡 DANGEROUS", len(dangerous), delta="LMTV / Reroute", delta_color="off")
c4.metric("🟢 SAFE (Staging Hubs)", len(safe), delta="Evacuation Zones", delta_color="normal")
c5.metric("👁️ Human Review", hitl_count, delta="Low Confidence Guard", delta_color="off")

st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Priority Triage Queue",
    "🚑 Smart Dispatch & Bilingual Alerts",
    "📊 Full CSV & JSON Reports",
    "📸 Live Image Triage & Tactical Q&A",
    "🔄 Before vs. After Change Detection",
])

# === TAB 1: PRIORITY TRIAGE QUEUE ===
with tab1:
    if not results:
        st.warning("No triage_report.json found in current directory.")
    else:
        col_f1, col_f2 = st.columns([3, 1])
        filter_status = col_f1.multiselect(
            "Filter by Threat Level:",
            options=["VERY DANGEROUS", "DANGEROUS", "SAFE"],
            default=["VERY DANGEROUS", "DANGEROUS", "SAFE"],
        )
        max_display = col_f2.selectbox("Max Cards Shown:", [25, 50, 100, 250, 1024], index=0)

        filtered = [r for r in results if r.get("status") in filter_status]
        st.caption(f"Showing top {min(len(filtered), max_display)} of {len(filtered)} matching sectors (ranked by Priority Score).")

        for item in filtered[:max_display]:
            status = item.get("status", "SAFE")
            badge = {"VERY DANGEROUS": "🔴", "DANGEROUS": "🟡", "SAFE": "🟢"}.get(status, "⚪")
            score = item.get("priority_score", 0)

            with st.container(border=True):
                c_img, c_info = st.columns([1, 2])
                img_path = Path(item["path"]) if item.get("path") else SERVER_DATASET_DIR / item["file"]
                if not img_path.exists():
                    img_path = SERVER_DATASET_DIR / item["file"]
                if img_path.exists():
                    c_img.image(str(img_path), use_container_width=True)

                with c_info:
                    st.subheader(f"{badge} {status} (Priority Score: {score}/100) — `{item['file']}`")
                    if item.get("needs_human_review"):
                        st.warning("⚠️ Low Model Confidence — Flagged for Human-in-the-Loop verification.")
                    st.caption(
                        f"📍 Sector: **{item.get('district')}** | 👥 Pop. Density: **{item.get('pop_density')}** | "
                        f"🏥 Visible Infra: **{item.get('critical_infra')}**"
                    )
                    st.write(f"**Water Classification:** `{item.get('water_type', 'N/A')}` | **Hazards:** {', '.join(item.get('hazards', []))}")
                    st.write(f"**Visual Evidence:** {item.get('evidence', '')}")
                    st.write("**Recommended Dispatch:** " + " • ".join([f"`{u}`" for u in item.get("dispatch_units", [])]))

# === TAB 2: SMART DISPATCH & REVERSE ROUTING ===
with tab2:
    st.subheader("Automated Evacuation Routing & Bilingual Citizen Alerts (WEA)")
    if not results:
        st.warning("No report data loaded.")
    else:
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("### 🔴 Top Critical Zones (VERY DANGEROUS)")
            for idx, item in enumerate(very_dangerous[:30]):
                assigned_hub = safe[idx % len(safe)]["file"] if safe else "External Dry Perimeter"
                with st.expander(f"[Score {item.get('priority_score')}/100] {item['file']} → Evacuate to {assigned_hub}", expanded=(idx < 5)):
                    st.error("⛔ REVERSE-TRIAGE BLOCK: Standard ambulances & light vehicles prohibited (submersion risk).")
                    st.write(f"**Visible Infrastructure:** {item.get('critical_infra')}")
                    st.write(f"**Deploy Units:** {', '.join(item.get('dispatch_units', []))}")
                    st.write(f"**Paired Safe Staging Hub:** `{assigned_hub}`")
                    st.code(
                        f"[WEA SMS - EN]: {item.get('citizen_alert_en', '')}\n"
                        f"[WEA SMS - ES]: {item.get('citizen_alert_es', '')}",
                        language="text",
                    )

        with col_r:
            st.markdown("### 🟡 Moderate Risk Zones (DANGEROUS)")
            for idx, item in enumerate(dangerous[:20]):
                with st.expander(f"[Score {item.get('priority_score')}/100] {item['file']} — Preventive Action", expanded=(idx < 3)):
                    st.warning("⚠️ Push flooded road coordinates to navigation APIs (Waze/Maps) for live rerouting.")
                    st.write(f"**Vulnerable Assets:** {item.get('critical_infra')}")
                    st.write(f"**Action Plan:** {', '.join(item.get('dispatch_units', []))}")

            st.markdown("### 🟢 Safe Logistics & Medical Hubs (SAFE)")
            if safe:
                st.success(f"Verified dry sectors available for field hospitals & supply depots: **{len(safe)}**")
                st.write(", ".join([f"`{s['file']}`" for s in safe[:40]]))

# === TAB 3: FULL CSV & JSON REPORTS ===
with tab3:
    st.subheader("Generated Dispatch Reports (`triage_report.csv` & `triage_report.json`)")
    if results:
        table_rows = [
            {
                "File": r.get("file"),
                "Status": r.get("status"),
                "Priority Score": r.get("priority_score"),
                "Confidence": r.get("confidence"),
                "Land Use": r.get("land_use"),
                "Water Type": r.get("water_type"),
                "Visible Infrastructure": r.get("critical_infra"),
                "Dispatch Units": ", ".join(r.get("dispatch_units", [])),
                "SMS Alert (EN)": r.get("citizen_alert_en"),
            }
            for r in results
        ]
        st.dataframe(table_rows, use_container_width=True, height=450)
        with st.expander("View Raw JSON Sample (Top 5 Priority Sectors)", expanded=False):
            st.json(results[:5])

# === TAB 4: LIVE SINGLE-IMAGE UPLOAD & TACTICAL Q&A ===
with tab4:
    st.subheader("Real-Time Drone / Aerial Feed Analysis & Tactical Q&A")
    uploaded = st.file_uploader("Upload an aerial image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="single_up")
    if uploaded:
        u1, u2 = st.columns([1, 1])
        u1.image(uploaded, caption=uploaded.name, use_container_width=True)
        with u2:
            if st.button("⚡ Run Instant AI Triage", type="primary"):
                with st.spinner("Qwen3-VL on NVIDIA GB10 is analyzing..."):
                    st.session_state["last_single_res"] = analyze_single_image(uploaded, uploaded.name)
            if "last_single_res" in st.session_state:
                r = st.session_state["last_single_res"]
                st.markdown(f"### {r['status']} — Priority Score: **{r['priority_score']}/100**")
                st.write(r["evidence"])
                st.info(f"📲 **SMS (EN):** {r['citizen_alert_en']}\n\n📲 **SMS (ES):** {r['citizen_alert_es']}")
            st.divider()
            custom_q = st.text_input("Ask a tactical question:", value="Can a high-clearance rescue truck pass the main road?")
            if st.button("Ask GB10") and custom_q:
                with st.spinner("Analyzing details..."):
                    st.success(ask_vlm(uploaded, custom_q))

# === TAB 5: BEFORE VS. AFTER ===
with tab5:
    st.subheader("Post-Disaster Damage Assessment: Before vs. After")
    cb, ca = st.columns(2)
    before_file = cb.file_uploader("1. Pre-Flood Baseline (Before)", type=["jpg", "jpeg", "png"], key="before")
    after_file = ca.file_uploader("2. Post-Flood Aerial (After)", type=["jpg", "jpeg", "png"], key="after")
    if before_file:
        cb.image(before_file, caption="BEFORE", use_container_width=True)
    if after_file:
        ca.image(after_file, caption="AFTER", use_container_width=True)
    if before_file and after_file and st.button("🔎 Compare Changes", type="primary"):
        with st.spinner("Comparing imagery on GB10..."):
            st.write(ask_vlm([before_file, after_file], COMPARE_PROMPT))
