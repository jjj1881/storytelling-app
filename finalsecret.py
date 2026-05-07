import asyncio
import hashlib
import os
import re
import tempfile
from io import BytesIO
from typing import Optional, Tuple

import streamlit as st
from PIL import Image
import torch
from transformers import BlipForConditionalGeneration, BlipProcessor
from openai import AzureOpenAI, OpenAI
from gtts import gTTS

try:
    import edge_tts
except Exception:
    edge_tts = None


# =========================
# App configuration
# =========================
st.set_page_config(
    page_title="Magic Storyteller for Kids",
    page_icon="📚",
    layout="wide",
)

APP_PASSWORD = "21264336"

CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"

DEFAULT_AZURE_ENDPOINT = "https://hkust.azure-api.net"
DEFAULT_AZURE_API_VERSION = "2024-02-01"
DEFAULT_AZURE_MODEL = "gpt-4o-mini"

# OpenAI TTS model and voice.
# If the same API key supports OpenAI's native audio endpoint, this will be used first.
# If not, the app automatically falls back to edge-tts / gTTS.
DEFAULT_OPENAI_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_OPENAI_TTS_VOICE = "nova"


# =========================
# Utility functions
# =========================
def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top left, #ffe6f2 0%, transparent 28%),
                    radial-gradient(circle at top right, #dff3ff 0%, transparent 30%),
                    linear-gradient(180deg, #fffaf0 0%, #f5f9ff 100%);
            }

            .hero-card {
                padding: 1.7rem 1.9rem;
                border-radius: 28px;
                background: linear-gradient(135deg, #fff1a8 0%, #ffd6ec 45%, #d8efff 100%);
                box-shadow: 0 14px 35px rgba(255, 154, 193, 0.28);
                margin-bottom: 1rem;
                border: 2px solid rgba(255,255,255,0.8);
            }

            .hero-card h1 {
                font-size: 2.4rem;
                color: #6347a3;
            }

            .soft-card {
                padding: 1rem 1.2rem;
                border-radius: 22px;
                background: rgba(255,255,255,.92);
                border: 1px solid rgba(255, 184, 216, .45);
                box-shadow: 0 8px 22px rgba(0,0,0,.05);
                margin-bottom: 0.85rem;
            }

            .story-box {
                padding: 1.25rem 1.35rem;
                border-radius: 22px;
                background: #fffef8;
                border: 2px dashed #ffc857;
                line-height: 1.85;
                font-size: 1.1rem;
                color: #3d3a4b;
                box-shadow: 0 8px 18px rgba(255, 200, 87, .18);
            }

            .small-note {
                font-size: .95rem;
                color: #5f5b75;
            }

            .step-done {
                padding:.75rem 1rem;
                border-radius:16px;
                background:#ecfff3;
                border:1px solid #b9efc9;
                margin:.5rem 0;
            }

            .step-active {
                padding:.75rem 1rem;
                border-radius:16px;
                background:#fff6d8;
                border:1px solid #f1d079;
                margin:.5rem 0;
            }

            .step-pending {
                padding:.75rem 1rem;
                border-radius:16px;
                background:#ffffff;
                border:1px solid #ececec;
                color:#777;
                margin:.5rem 0;
            }

            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #fff4fb 0%, #eef8ff 100%);
            }

            .stButton > button {
                border-radius: 999px;
                font-weight: 700;
            }

            .stDownloadButton > button {
                border-radius: 999px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text or ""))


def image_hash(image_bytes: bytes) -> str:
    return hashlib.md5(image_bytes).hexdigest()


def get_secret_or_env(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, "")
        if value:
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


def first_available_secret_or_env(names: list[str], default: str = "") -> str:
    for name in names:
        value = get_secret_or_env(name, "")
        if value:
            return value
    return default


# =========================
# Password gate
# =========================
def check_password() -> bool:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    inject_custom_css()

    st.markdown(
        """
        <div class="hero-card">
            <h1 style="margin-bottom:0.4rem;">🔐 Magic Story World</h1>
            <p style="font-size:1.1rem; margin-bottom:0;">
                Enter the magic password to open the storybook.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    password = st.text_input(
        "Magic password",
        type="password",
        placeholder="Enter password",
    )

    if st.button("🌈 Enter Story World", type="primary", use_container_width=True):
        if password == APP_PASSWORD:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Oops! The magic password is not correct.")

    return False


# =========================
# Model loading
# =========================
@st.cache_resource(show_spinner=False)
def load_caption_model():
    processor = BlipProcessor.from_pretrained(CAPTION_MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_NAME)
    model.eval()
    return processor, model


# =========================
# Caption generation
# =========================
def clean_caption(caption: str) -> str:
    caption = normalize_text(caption)
    caption = re.sub(
        r"^(a picture of|an image of|there is|there are)\s+",
        "",
        caption,
        flags=re.I,
    )
    caption = caption.strip(" .,;:")

    if not caption or count_words(caption) < 3:
        return "a friendly child discovering something interesting in a bright place"

    return caption


def generate_caption(image: Image.Image) -> str:
    processor, model = load_caption_model()

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=40,
            num_beams=5,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )

    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return clean_caption(caption)


# =========================
# Story generation
# =========================
def build_story_messages(caption: str, story_style: str, child_name: str) -> list[dict]:
    hero_name = child_name.strip() if child_name.strip() else "the child"

    system_message = (
        "You are a professional children's storyteller. "
        "You write safe, warm, coherent stories for children aged 3 to 10. "
        "Use simple, natural English. Do not mention AI, prompts, captions, or images. "
        "Avoid scary, violent, unsafe, or adult content."
    )

    user_message = f"""
Write ONE complete short children's story based on the visual scene below.

Visual scene:
{caption}

Story settings:
- Main character: {hero_name}
- Style: {story_style}
- Length: 50 to 100 words
- Audience: children aged 3 to 10

The story must include:
1. A clear beginning
2. One small problem, question, or surprise
3. A kind action or solution
4. A happy ending

Rules:
- Output only the story.
- No title.
- No bullet points.
- No explanation.
- Keep the language simple and smooth.
""".strip()

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


def clean_story_text(text: str) -> str:
    text = normalize_text(text)
    text = text.strip().strip('"').strip("'")

    text = re.sub(r"^(Story:|Title:.*?\n)", "", text, flags=re.I | re.S).strip()

    stop_patterns = [
        "\nExplanation:",
        "\nNote:",
        "\nRequirements:",
        "\nImage:",
        "\nCaption:",
        "Explanation:",
        "Requirements:",
        "Caption:",
        "Prompt:",
        "Here is",
    ]

    for pattern in stop_patterns:
        idx = text.find(pattern)
        if idx > 20:
            text = text[:idx].strip()

    sentences = re.split(r"(?<=[.!?])\s+", text)
    kept = []

    for sentence in sentences:
        sentence = normalize_text(sentence)
        if not sentence:
            continue

        if count_words(" ".join(kept + [sentence])) <= 100:
            kept.append(sentence)

    story = " ".join(kept).strip() if kept else text

    if count_words(story) > 100:
        story = " ".join(story.split()[:100]).rstrip(" ,;:") + "."

    if story and story[-1] not in ".!?":
        story += "."

    return story


def story_quality_check(story: str) -> tuple[bool, str]:
    words = count_words(story)

    if words < 50:
        return False, "The story is shorter than 50 words."

    if words > 100:
        return False, "The story is longer than 100 words."

    lower = story.lower()
    banned = [
        "caption",
        "image",
        "prompt",
        "as an ai",
        "requirements",
        "bullet point",
        "kill",
        "blood",
        "dead",
        "weapon",
        "monster ate",
    ]

    if any(x in lower for x in banned):
        return False, "The story contains unwanted meta or unsafe wording."

    sentence_count = sum(story.count(mark) for mark in ".!?")

    if sentence_count < 3:
        return False, "The story has too few complete sentences."

    return True, "OK"


def create_azure_client(api_key: str, endpoint: str, api_version: str) -> AzureOpenAI:
    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint.rstrip("/"),
        api_version=api_version,
        default_headers={"Ocp-Apim-Subscription-Key": api_key},
    )


def call_azure_story_model(
    messages: list[dict],
    model_name: str,
    api_key: str,
    endpoint: str,
    api_version: str,
    temperature: float = 0.75,
) -> str:
    client = create_azure_client(api_key, endpoint, api_version)

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=180,
    )

    return response.choices[0].message.content or ""


def generate_story(
    caption: str,
    story_style: str,
    child_name: str,
    model_name: str,
    api_key: str,
    endpoint: str,
    api_version: str,
) -> str:
    messages = build_story_messages(caption, story_style, child_name)

    raw_story = call_azure_story_model(
        messages=messages,
        model_name=model_name,
        api_key=api_key,
        endpoint=endpoint,
        api_version=api_version,
        temperature=0.75,
    )

    story = clean_story_text(raw_story)
    ok, reason = story_quality_check(story)

    if ok:
        return story

    retry_messages = messages + [
        {"role": "assistant", "content": story},
        {
            "role": "user",
            "content": (
                f"Please rewrite the story. Problem: {reason} "
                "Return only one coherent children's story, 50 to 100 words, "
                "with a clear beginning, a small problem, a kind solution, and a happy ending."
            ),
        },
    ]

    raw_story = call_azure_story_model(
        messages=retry_messages,
        model_name=model_name,
        api_key=api_key,
        endpoint=endpoint,
        api_version=api_version,
        temperature=0.60,
    )

    story = clean_story_text(raw_story)

    if count_words(story) > 100:
        story = " ".join(story.split()[:100]).rstrip(" ,;:") + "."

    return story


# =========================
# Text-to-speech
# =========================
def build_tts_narration_text(story: str, story_style: str) -> str:
    """
    Keep the output text natural for TTS.
    OpenAI TTS handles emotion better when the input includes a short narration instruction.
    The instruction is not shown in the UI; it only affects the generated voice.
    """
    style_instruction = {
        "Bedtime": "Read this as a warm, slow bedtime story for a young child. Use a soft, gentle, soothing tone with natural pauses.",
        "Adventure": "Read this as an exciting but child-safe adventure story. Use an energetic, cheerful tone with expressive pacing.",
        "Magic": "Read this as a magical children's story. Use a warm, curious, playful voice with gentle wonder.",
        "Friendship": "Read this as a kind friendship story for children. Use a friendly, caring, natural voice.",
    }.get(
        story_style,
        "Read this as a warm children's story. Use a natural, friendly, expressive voice.",
    )

    return f"{style_instruction}\n\n{normalize_text(story)}"


def openai_tts_to_bytes(
    text: str,
    story_style: str,
    api_key: str,
    tts_model: str,
    tts_voice: str,
) -> bytes:
    """
    OpenAI native TTS.
    This requires a normal OpenAI API key that supports the audio speech endpoint.
    If the current key is only an Azure/HKUST gateway key, this function may fail;
    the caller will fall back to edge-tts or gTTS automatically.
    """
    client = OpenAI(api_key=api_key)
    narration_text = build_tts_narration_text(text, story_style)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_path = temp_file.name

    try:
        response = client.audio.speech.create(
            model=tts_model,
            voice=tts_voice,
            input=narration_text,
            response_format="mp3",
        )

        response.stream_to_file(temp_path)

        with open(temp_path, "rb") as audio_file:
            return audio_file.read()

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def voice_settings(story_style: str, slow_reading: bool) -> Tuple[str, str, str]:
    if story_style == "Bedtime":
        voice = "en-US-JennyNeural"
        rate = "-22%" if slow_reading else "-14%"
        pitch = "-2Hz"
    elif story_style == "Adventure":
        voice = "en-US-GuyNeural"
        rate = "-10%" if slow_reading else "-3%"
        pitch = "+3Hz"
    elif story_style == "Magic":
        voice = "en-US-AriaNeural"
        rate = "-14%" if slow_reading else "-6%"
        pitch = "+5Hz"
    else:
        voice = "en-US-JennyNeural"
        rate = "-14%" if slow_reading else "-6%"
        pitch = "+2Hz"

    return voice, rate, pitch


async def edge_tts_to_bytes(text: str, story_style: str, slow_reading: bool) -> bytes:
    voice, rate, pitch = voice_settings(story_style, slow_reading)

    communicate = edge_tts.Communicate(
        text=normalize_text(text),
        voice=voice,
        rate=rate,
        pitch=pitch,
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_path = temp_file.name

    try:
        await communicate.save(temp_path)

        with open(temp_path, "rb") as audio_file:
            return audio_file.read()

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def gtts_to_bytes(text: str, slow_reading: bool) -> bytes:
    tts = gTTS(text=text, lang="en", slow=slow_reading)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        temp_path = temp_file.name

    try:
        tts.save(temp_path)

        with open(temp_path, "rb") as audio_file:
            return audio_file.read()

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def text_to_speech(
    story: str,
    story_style: str,
    api_key: str,
    slow_reading: bool = False,
    tts_model: str = DEFAULT_OPENAI_TTS_MODEL,
    tts_voice: str = DEFAULT_OPENAI_TTS_VOICE,
) -> tuple[Optional[bytes], str]:
    """
    Priority:
    1. OpenAI native TTS, using the same API key if it supports the audio endpoint.
    2. edge-tts neural voice fallback.
    3. gTTS fallback.
    """
    if api_key:
        try:
            audio = openai_tts_to_bytes(
                text=story,
                story_style=story_style,
                api_key=api_key,
                tts_model=tts_model,
                tts_voice=tts_voice,
            )
            return audio, "OpenAI expressive storytelling voice"
        except Exception:
            pass

    if edge_tts is not None:
        try:
            audio = asyncio.run(edge_tts_to_bytes(story, story_style, slow_reading))
            return audio, "Warm neural voice"
        except Exception:
            pass

    try:
        return gtts_to_bytes(story, slow_reading), "Gentle fallback voice"
    except Exception:
        return None, "Audio generation failed"


# =========================
# UI rendering
# =========================
def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <h1 style="margin-bottom:0.35rem;">📚 Magic Storyteller for Kids</h1>
            <p style="font-size:1.08rem; margin-bottom:0.35rem;">
                Upload a picture and watch it become a tiny magical story.
            </p>
            <p class="small-note" style="margin-bottom:0;">
                A colorful storybook app for children aged 3–10.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[str, str, bool, str, str, str, str, str, str]:
    st.sidebar.markdown("## 🌈 Story Magic")

    child_name = st.sidebar.text_input(
        "Little hero's name",
        placeholder="e.g. Emma",
    )

    story_style = st.sidebar.selectbox(
        "Story style",
        ["Adventure", "Magic", "Friendship", "Bedtime"],
        index=1,
    )

    slow_reading = st.sidebar.checkbox(
        "Gentle slow reading",
        value=True,
    )

    model_name = get_secret_or_env("AZURE_OPENAI_MODEL", DEFAULT_AZURE_MODEL).strip()
    endpoint = get_secret_or_env("AZURE_OPENAI_ENDPOINT", DEFAULT_AZURE_ENDPOINT).strip()
    api_version = get_secret_or_env(
        "AZURE_OPENAI_API_VERSION",
        DEFAULT_AZURE_API_VERSION,
    ).strip()

    api_key = first_available_secret_or_env(
        [
            "AZURE_OPENAI_API_KEY",
            "HKUST_AZURE_OPENAI_API_KEY",
            "OPENAI_API_KEY",
        ],
        "",
    ).strip()

    tts_model = get_secret_or_env("OPENAI_TTS_MODEL", DEFAULT_OPENAI_TTS_MODEL).strip()
    tts_voice = get_secret_or_env("OPENAI_TTS_VOICE", DEFAULT_OPENAI_TTS_VOICE).strip()

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="font-size:0.98rem; line-height:1.7;">
        🧸 Upload a picture<br>
        ✨ Create a short story<br>
        🔊 Listen to the story<br>
        🌙 Enjoy a gentle ending
        </div>
        """,
        unsafe_allow_html=True,
    )

    return (
        child_name,
        story_style,
        slow_reading,
        model_name,
        api_key,
        endpoint,
        api_version,
        tts_model,
        tts_voice,
    )


def render_progress(stage: str) -> None:
    states = {
        "captioning": ["active", "pending", "pending"],
        "story": ["done", "active", "pending"],
        "audio": ["done", "done", "active"],
        "done": ["done", "done", "done"],
        "idle": ["pending", "pending", "pending"],
    }.get(stage, ["pending", "pending", "pending"])

    labels = [
        "1. Looking at the picture",
        "2. Writing the story",
        "3. Making the audio",
    ]

    icons = {
        "done": "✅ Done",
        "active": "⏳ Creating magic",
        "pending": "⬜ Waiting",
    }

    classes = {
        "done": "step-done",
        "active": "step-active",
        "pending": "step-pending",
    }

    for label, state in zip(labels, states):
        st.markdown(
            f"<div class='{classes[state]}'><b>{label}</b>&nbsp;&nbsp; {icons[state]}</div>",
            unsafe_allow_html=True,
        )


def render_live_results(result: dict) -> None:
    st.markdown("<div class='soft-card'>", unsafe_allow_html=True)

    status = result.get("status", "idle")

    if status == "done":
        st.write("**Your magical story is ready! 🌟**")
    elif status == "captioning":
        st.write("**Looking carefully at the picture... 🖼️**")
    elif status == "story":
        st.write("**Writing a warm little story... ✨**")
    elif status == "audio":
        st.write("**Preparing a gentle voice... 🔊**")
    else:
        st.write("Upload a picture and click **Generate my story**.")

    st.markdown("</div>", unsafe_allow_html=True)

    render_progress(status)

    caption = result.get("caption")
    story = result.get("story")
    audio = result.get("audio")
    voice_label = result.get("voice_label")

    if "render_sequence" not in st.session_state:
        st.session_state.render_sequence = 0

    st.session_state.render_sequence += 1
    render_id = st.session_state.render_sequence

    result_id = f"{result.get('image_hash', 'result')}_{result.get('run_id', '0')}_{render_id}"

    if caption:
        st.subheader("🖼️ What I found")
        st.markdown(f"<div class='soft-card'>{caption}</div>", unsafe_allow_html=True)

    if story:
        st.subheader("🌟 Your story")
        st.markdown(f"<div class='story-box'>{story}</div>", unsafe_allow_html=True)

        words = count_words(story)
        st.caption(f"Word count: {words} words")

        if not (50 <= words <= 100):
            st.warning("The story should be 50–100 words. Try generating again if needed.")

        if status == "done":
            st.download_button(
                "⬇️ Download story",
                data=story,
                file_name="generated_story.txt",
                mime="text/plain",
                use_container_width=True,
                key=f"download_story_{result_id}",
            )

    if status == "audio" and not audio:
        st.subheader("🔊 Listen to the story")
        st.info("The story voice is being prepared...")

    if audio:
        st.subheader("🔊 Listen to the story")

        if voice_label:
            st.caption(f"Voice: {voice_label}")

        st.audio(audio, format="audio/mp3")

        if status == "done":
            st.download_button(
                "⬇️ Download audio",
                data=audio,
                file_name="story_audio.mp3",
                mime="audio/mpeg",
                use_container_width=True,
                key=f"download_audio_{result_id}",
            )


# =========================
# Main app
# =========================
def main() -> None:
    if not check_password():
        return

    inject_custom_css()
    render_header()

    (
        child_name,
        story_style,
        slow_reading,
        model_name,
        api_key,
        endpoint,
        api_version,
        tts_model,
        tts_voice,
    ) = render_sidebar()

    if "result" not in st.session_state:
        st.session_state.result = {"status": "idle"}

    left_col, right_col = st.columns([1.15, 0.85], gap="large")

    with left_col:
        st.subheader("1️⃣ Upload a picture")

        uploaded_file = st.file_uploader(
            "Choose a picture",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        current_hash = None
        image = None

        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            current_hash = image_hash(image_bytes)

            image = Image.open(BytesIO(image_bytes)).convert("RGB")

            st.image(
                image,
                caption="Your picture",
                use_container_width=True,
            )

            if st.session_state.result.get("image_hash") not in (None, current_hash):
                st.session_state.result = {
                    "status": "idle",
                    "image_hash": current_hash,
                }

            generate_clicked = st.button(
                "✨ Generate my story",
                type="primary",
                use_container_width=True,
            )

        else:
            generate_clicked = False
            st.info("Upload an image to begin the story magic.")

    with right_col:
        st.subheader("2️⃣ Story result")

        result_placeholder = st.empty()

        with result_placeholder.container():
            render_live_results(st.session_state.result)

    if generate_clicked and image is not None and current_hash is not None:
        if not api_key:
            st.error(
                "The story service is not configured yet. "
                "Please add AZURE_OPENAI_API_KEY, HKUST_AZURE_OPENAI_API_KEY, or OPENAI_API_KEY to Streamlit Cloud secrets."
            )
            return

        try:
            if "generation_counter" not in st.session_state:
                st.session_state.generation_counter = 0

            st.session_state.generation_counter += 1
            current_run_id = st.session_state.generation_counter

            st.session_state.result = {
                "status": "captioning",
                "image_hash": current_hash,
                "run_id": current_run_id,
            }

            with result_placeholder.container():
                render_live_results(st.session_state.result)

            caption = generate_caption(image)

            st.session_state.result.update(
                {
                    "caption": caption,
                    "status": "story",
                }
            )

            with result_placeholder.container():
                render_live_results(st.session_state.result)

            story = generate_story(
                caption=caption,
                story_style=story_style,
                child_name=child_name,
                model_name=model_name,
                api_key=api_key,
                endpoint=endpoint,
                api_version=api_version,
            )

            st.session_state.result.update(
                {
                    "story": story,
                    "status": "audio",
                }
            )

            with result_placeholder.container():
                render_live_results(st.session_state.result)

            audio, voice_label = text_to_speech(
                story=story,
                story_style=story_style,
                api_key=api_key,
                slow_reading=slow_reading,
                tts_model=tts_model,
                tts_voice=tts_voice,
            )

            st.session_state.result.update(
                {
                    "audio": audio,
                    "voice_label": voice_label,
                    "status": "done",
                }
            )

            with result_placeholder.container():
                render_live_results(st.session_state.result)

            return

        except Exception as error:
            st.error("Something went wrong while generating the story.")
            st.exception(error)

    st.markdown("---")
    st.caption("Made for a warm and playful storytelling experience.")


if __name__ == "__main__":
    main()
