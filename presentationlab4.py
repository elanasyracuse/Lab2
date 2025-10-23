# lab6_fact_checker.py — AI Fact-Checker + Citation Builder
# Streamlit + OpenAI Responses API with web_search tool
# -----------------------------------------------------
# Prereqs:
#   pip install streamlit openai
# Secrets:
#   Put OPENAI_API_KEY in .streamlit/secrets.toml or your environment.

import os
import json
import time
from typing import Dict, Any, List, Tuple

import streamlit as st
from openai import OpenAI

# ================================
# Page setup & keys
# ================================
st.set_page_config(page_title="AI Fact-Checker + Citation Builder", page_icon="🕵️‍♀️", layout="centered")
st.title("🕵️‍♀️ AI Fact-Checker + Citation Builder")

# Same secrets pattern as your Lab 4
OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
if not OPENAI_KEY:
    st.warning("Set your OPENAI_API_KEY in .streamlit/secrets.toml or environment.")

client = OpenAI(api_key=OPENAI_KEY)

# ================================
# Helpers
# ================================
SYSTEM_PROMPT = """You are an evidence-based fact-checking assistant.
Your job: verify a single factual claim using live search, then return a STRICT JSON object with NO additional text before or after.

You MUST return ONLY this exact JSON structure:
{
  "claim": "the claim being verified",
  "verdict": "True OR False OR Mixed OR Uncertain",
  "explanation": "detailed explanation of the verdict based on evidence",
  "confidence": 0.85,
  "sources": [
    {
      "title": "source title",
      "url": "https://source-url.com",
      "quote_or_evidence": "relevant quote or evidence from this source"
    }
  ]
}

Rules:
- ALWAYS use the web_search tool to verify; do not rely on prior knowledge alone.
- Prefer primary sources (gov/edu/academic/reputable orgs) and recency when timeliness matters.
- If evidence conflicts, choose 'Mixed' or 'Uncertain' and explain why.
- Include at least 1 source, preferably 3-5 sources.
- Confidence should be a number between 0 and 1.
- CRITICAL: Return ONLY the JSON object. No markdown, no code blocks, no extra text.
"""


def _mk_clickable_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for s in sources:
        title = (s.get("title") or "Source").strip()
        url = (s.get("url") or "").strip()
        link = f"[{title}]({url})" if url else title
        out.append({**s, "link": link})
    return out

def _basic_confidence_hint(sources: List[Dict[str, Any]]) -> str:
    n = len(sources or [])
    if n >= 4: return "High: multiple independent sources."
    if n == 3: return "Medium: several sources."
    return "Low: limited sourcing."

def fact_check_claim(user_claim: str) -> Tuple[Dict[str, Any], str]:
    """Call OpenAI /v1/responses with web_search tool. Returns (parsed_json, raw_text)."""
    if not user_claim or not user_claim.strip():
        raise ValueError("Please provide a non-empty claim.")

    resp = client.responses.create(
        model="gpt-4.1-mini",  # or "gpt-4.1" / "gpt-4o-mini"
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_claim.strip()},
        ],
        tools=[
            {"type": "web_search"}
        ],
        temperature=0.2,
    )

    # Parse the response
    raw_text = ""
    try:
        # Try to get output_text first
        if hasattr(resp, "output_text") and resp.output_text:
            raw_text = resp.output_text
        # Try output.content structure
        elif hasattr(resp, "output") and resp.output:
            if hasattr(resp.output, "content"):
                for c in resp.output.content:
                    if hasattr(c, "type") and c.type == "output_text" and hasattr(c, "text"):
                        raw_text += c.text
            # Fallback to direct access
            elif isinstance(resp.output, list) and len(resp.output) > 0:
                if hasattr(resp.output[0], "content") and len(resp.output[0].content) > 0:
                    raw_text = resp.output[0].content[0].text
    except Exception as e:
        raise RuntimeError(f"Could not extract response: {e}")

    if not raw_text:
        raise RuntimeError("The model returned an empty response.")

    # Clean up the response - remove markdown code blocks if present
    raw_text = raw_text.strip()
    if raw_text.startswith("```json"):
        raw_text = raw_text[7:]
    elif raw_text.startswith("```"):
        raw_text = raw_text[3:]
    if raw_text.endswith("```"):
        raw_text = raw_text[:-3]
    raw_text = raw_text.strip()

    # Parse JSON (with salvage fallback)
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                parsed = json.loads(raw_text[start:end])
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Could not parse JSON from response: {e}\nRaw text: {raw_text}")
        else:
            raise RuntimeError(f"No JSON object found in response.\nRaw text: {raw_text}")

    # Validate required fields
    required_fields = ["claim", "verdict", "explanation", "confidence", "sources"]
    missing = [f for f in required_fields if f not in parsed]
    if missing:
        raise RuntimeError(f"Response missing required fields: {missing}\nParsed: {parsed}")

    return parsed, raw_text

# ================================
# Session state
# ================================
if "fc_history" not in st.session_state:
    st.session_state.fc_history = []  # {claim, verdict, confidence, time, sources, obj}

# ================================
# UI — Input
# ================================
st.subheader("Check a claim")
user_claim = st.text_input(
    "Enter a factual claim:",
    placeholder='e.g., "Dark chocolate is healthy."',
    key="user_claim_value"
)

c1, c2 = st.columns([1, 1])
with c1:
    go = st.button("Check Fact", type="primary")
with c2:
    st.caption("Uses OpenAI Responses API + live web search to verify and cite sources.")

if go:
    if not OPENAI_KEY:
        st.error("Missing OPENAI_API_KEY. Add it to secrets or env and try again.")
    elif not st.session_state.user_claim_value.strip():
        st.warning("Please enter a claim.")
    else:
        with st.spinner("Verifying with live sources…"):
            try:
                result_obj, _ = fact_check_claim(st.session_state.user_claim_value)
                result_obj["sources"] = _mk_clickable_sources(result_obj.get("sources", []))
                st.session_state.fc_history.insert(0, {
                    "claim": result_obj.get("claim", st.session_state.user_claim_value),
                    "verdict": result_obj.get("verdict", "Uncertain"),
                    "confidence": float(result_obj.get("confidence", 0)),
                    "time": time.strftime("%Y-%m-%d %H:%M"),
                    "sources": result_obj.get("sources", []),
                    "obj": result_obj
                })
            except Exception as e:
                st.error(f"Error: {e}")

st.divider()

# ================================
# Results (latest)
# ================================
if st.session_state.fc_history:
    latest = st.session_state.fc_history[0]["obj"]
    st.markdown("### Latest Result")
    verdict = latest.get("verdict", "Uncertain")
    conf = float(latest.get("confidence", 0.0))
    st.markdown(f"**Verdict:** `{verdict}`  |  **Confidence:** `{conf:.2f}`  ·  {_basic_confidence_hint(latest.get('sources', []))}")

    with st.expander("Sources (click to open)"):
        for i, s in enumerate(latest.get("sources", []), start=1):
            st.markdown(f"**{i}.** {s.get('link', s.get('title','Source'))}\n\n> {s.get('quote_or_evidence','')}\n")

    st.markdown("**Structured JSON**")
    st.json(latest)

# ================================
# History list
# ================================
st.divider()
st.markdown("### History")
if not st.session_state.fc_history:
    st.caption("No checks yet. Try one of the samples below.")
else:
    for item in st.session_state.fc_history[:5]:
        st.markdown(f"- **{item['claim']}** → `{item['verdict']}` · conf `{item['confidence']:.2f}` · _{item['time']}_")

# ================================
# Samples
# ================================

def set_claim_text(sample_text: str):
    """Callback to update the text input field."""
    st.session_state.user_claim_value = sample_text

with st.expander("Try sample claims"):
    cols = st.columns(3)
    samples = [
        "Dark chocolate is healthy.",
        "Coffee prevents cancer.",
        "Pluto is still classified as a planet."
    ]
    for idx, s in enumerate(samples):
        with cols[idx]:
            st.button(
                s,
                key=f"sample_{idx}",
                on_click=set_claim_text,
                args=(s,)
            )

# ================================
# Reflection
# ================================
st.divider()
st.subheader("Reflection")
st.markdown("""
- **Structured reasoning:** Strong system prompt enforces verdict, explanation, confidence, and sources structure.
- **Verifiability:** `web_search` tool returns clickable citations so you can inspect evidence.
- **Confidence hint:** A quick heuristic based on number of independent sources cited.
""")