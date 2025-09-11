# lab3.py
import os, time, sys, traceback, re
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from openai import APIError, RateLimitError, APITimeoutError, APIConnectionError

# --- Setup ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY is missing. Add it to your .env file.")
    st.stop()

client = OpenAI(api_key=api_key)

st.title("💬 Lab 3A – Kid-Friendly Chatbot")

# ---- Config ----
MODEL = "gpt-4o-mini"
BUFFER_EXCHANGES = 20  # keep last 20 user↔assistant exchanges (40 msgs)
SYSTEM_PROMPT = (
    "You are a kind teaching assistant. Explain things so a 10-year-old can understand. "
    "Use short sentences, simple words, and a friendly tone. "
    "Give tiny examples when helpful. If unsure, say you’re not sure."
)
FOLLOW_UP_PROMPT = "Do you want more information?"

# ---- Session State ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list[{"role": "...", "content": "..."}]
if "last_user_question" not in st.session_state:
    st.session_state.last_user_question = None  # track the last topic

# ---- UI: show history ----
for m in st.session_state.chat_history:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---- Intent helpers ----
GREETING_PAT = re.compile(
    r"^\s*(hi|hello|hey|howdy|good\s+(morning|afternoon|evening)|"
    r"(i\s*'?m|i\s*am)\s+\w+|my\s+name\s+is\s+\w+|nice\s+to\s+meet\s+you)\b",
    re.IGNORECASE
)

ACK_PAT = re.compile(
    r"^\s*(thanks|thank you|got it|cool|great|awesome|ok|okay|all good)\b[.! ]*$",
    re.IGNORECASE
)

def is_greeting(text: str) -> bool:
    return bool(GREETING_PAT.match(text.strip()))

def is_ack(text: str) -> bool:
    return bool(ACK_PAT.match(text.strip()))

def is_yes(text: str) -> bool:
    t = text.strip().lower()
    return bool(re.fullmatch(r"(y|yes|yeah|yep|sure|ok|okay|please|more|more info|tell me more)\.?", t))

def is_no(text: str) -> bool:
    t = text.strip().lower()
    return bool(re.fullmatch(r"(n|no|nope|nah|not now|i'?m good|im good)\.?", t))

def is_question(text: str) -> bool:
    t = text.strip().lower()
    return (
        t.endswith("?")
        or t.startswith((
            "what", "why", "how", "when", "where", "who",
            "can", "could", "should", "explain", "tell me", "help me understand"
        ))
    )

def is_topic(text: str) -> bool:
    """
    A 'topic' is anything the user asks/tells that seeks info or help,
    not a greeting, not yes/no, not a quick acknowledgement.
    We treat any non-greeting, non-yes/no, non-ack message as a topic.
    """
    t = text.strip()
    if not t:
        return False
    if is_greeting(t) or is_yes(t) or is_no(t) or is_ack(t):
        return False
    return True  # broad on purpose so we follow up after each topic the user asks

# ---- Message building and streaming ----
def build_payload_messages(pending_user_message: str | None = None):
    trimmed = st.session_state.chat_history[-(BUFFER_EXCHANGES * 2):]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + trimmed
    if pending_user_message is not None:
        messages.append({"role": "user", "content": pending_user_message})
    return messages

def generate_response_streaming(messages):
    delays = [0.8, 1.6, 3.2]
    last_err = None
    for delay in [0] + delays:
        if delay:
            time.sleep(delay)
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=messages,
                stream=True,
            )
        except (RateLimitError, APITimeoutError, APIConnectionError, APIError) as e:
            last_err = e
            continue
    try:
        comp = client.chat.completions.create(model=MODEL, messages=messages, stream=False)
        class OneShot:
            def __iter__(self_inner):
                content = comp.choices[0].message.content or ""
                yield type("Chunk", (), {"choices":[type("Ch", (), {"delta": type("D", (), {"content": content})()})()]})()
        return OneShot()
    except Exception as e2:
        raise last_err or e2

def stream_and_render(payload_messages, append_follow_up: bool):
    """Streams a reply. Optionally appends the follow-up question line."""
    placeholder = st.empty()
    full = ""
    try:
        stream = generate_response_streaming(payload_messages)
        for chunk in stream:
            delta = getattr(chunk.choices[0], "delta", None)
            piece = getattr(delta, "content", None) if delta else None
            if piece:
                full += piece
                placeholder.markdown(full + "▌")
        if append_follow_up and full.strip():
            full = (full.strip() + "\n\n" + FOLLOW_UP_PROMPT).strip()
        placeholder.markdown(full or "_(no content)_")
    except Exception as e:
        full = ""
        placeholder.markdown(":warning: I hit a server hiccup. Try again in a moment.")
        with st.expander("Technical details"):
            st.code("".join(traceback.format_exception_only(type(e), e)).strip())
    return full

# ---- Chat input ----
if prompt := st.chat_input("Say hi or ask a question!"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # YES → expand the last topic and re-ask follow-up
    if is_yes(prompt):
        if st.session_state.last_user_question:
            with st.chat_message("assistant"):
                expand_instruction = (
                    "The user said YES to more info. "
                    "Please add more helpful details about your last answer to this topic: "
                    f"\"{st.session_state.last_user_question}\". "
                    "Keep it simple for a 10-year-old. Use short sentences and tiny examples."
                )
                payload = build_payload_messages(pending_user_message=expand_instruction)
                full = stream_and_render(payload, append_follow_up=True)
            st.session_state.chat_history.append({"role": "assistant", "content": full})
        else:
            with st.chat_message("assistant"):
                msg = "Sure! Tell me the topic you want to learn about."
                st.markdown(msg)
            st.session_state.chat_history.append({"role": "assistant", "content": msg})

    # NO → ask what else to help with (no follow-up)
    elif is_no(prompt):
        with st.chat_message("assistant"):
            msg = "Okay! What can I help you with?"
            st.markdown(msg)
        st.session_state.chat_history.append({"role": "assistant", "content": msg})

    # Greeting → friendly welcome (no follow-up)
    elif is_greeting(prompt):
        st.session_state.last_user_question = None
        with st.chat_message("assistant"):
            msg = "Hi! It’s nice to meet you! How can I help you today?"
            st.markdown(msg)
        st.session_state.chat_history.append({"role": "assistant", "content": msg})

    # Acknowledgement → polite close (no follow-up)
    elif is_ack(prompt):
        with st.chat_message("assistant"):
            msg = "You’re welcome! Want to learn something else?"
            st.markdown(msg)
        st.session_state.chat_history.append({"role": "assistant", "content": msg})

        # (Optional) don’t mark this as a topic; keep last topic for a possible later "yes"

    # Topic (question or request) → answer and follow up
    else:
        st.session_state.last_user_question = prompt  # remember the topic
        with st.chat_message("assistant"):
            payload = build_payload_messages(pending_user_message=None)  # user already appended
            # append follow-up because this is a topic
            full = stream_and_render(payload, append_follow_up=True)
        st.session_state.chat_history.append({"role": "assistant", "content": full})

    # Trim to last 20 exchanges (40 messages)
    if len(st.session_state.chat_history) > BUFFER_EXCHANGES * 2:
        st.session_state.chat_history = st.session_state.chat_history[-(BUFFER_EXCHANGES * 2):]
