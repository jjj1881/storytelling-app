"""
Little Dream Storyteller

A Streamlit storytelling application for children aged 3-10.
Users upload an image, the app creates a short 50-100 word story from the image,
and the story is converted into audio narration.

"""

from __future__ import annotations

import hashlib
import re
from io import BytesIO
from typing import Any, Optional

import streamlit as st
from gtts import gTTS
from PIL import Image
from transformers import pipeline


# =============================================================================
# 1. App configuration
# =============================================================================

st.set_page_config(
    page_title="Little Dream Storyteller",
    page_icon="🌈",
    layout="wide",
    initial_sidebar_state="expanded",
)

IMAGE_CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
STORY_GENERATION_MODEL = "distilgpt2"

MIN_STORY_WORDS = 50
MAX_STORY_WORDS = 100

STORY_STYLES = {
    "Magic Forest ✨": {
        "style_key": "Magic",
        "description": "gentle, imaginative, bright, and full of wonder",
        "button_hint": "Add a sprinkle of magic to the story.",
    },
    "Tiny Adventure 🚀": {
        "style_key": "Adventure",
        "description": "brave, playful, energetic, and safe",
        "button_hint": "Make the story active and curious.",
    },
    "Kind Friends 🧸": {
        "style_key": "Friendship",
        "description": "warm, caring, kind, and cooperative",
        "button_hint": "Focus on helping and sharing.",
    },
    "Sleepy Stars 🌙": {
        "style_key": "Bedtime",
        "description": "soft, calm, comforting, and peaceful",
        "button_hint": "Create a gentle bedtime feeling.",
    },
}

READING_SPEEDS = {
    "Gentle and slow 🐢": True,
    "Clear and normal 🐇": False,
}


# =============================================================================
# 2. Styling and UI helper functions
# =============================================================================

def inject_custom_css() -> None:
    """Inject custom CSS for a playful, child-friendly Streamlit interface."""
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at 8% 10%, #ffe3f3 0%, transparent 28%),
                    radial-gradient(circle at 92% 8%, #d8f5ff 0%, transparent 31%),
                    radial-gradient(circle at 50% 100%, #fff1bd 0%, transparent 34%),
                    linear-gradient(180deg, #fffdf7 0%, #f4fbff 100%);
            }

            .block-container {
                padding-top: 1.3rem;
                padding-bottom: 2rem;
            }

            .hero-card {
                padding: 1.9rem 2.1rem;
                border-radius: 34px;
                background: linear-gradient(135deg, #fff1a8 0%, #ffd4ea 45%, #d8f1ff 100%);
                box-shadow: 0 18px 45px rgba(116, 92, 199, 0.18);
                border: 2px solid rgba(255,255,255,0.95);
                margin-bottom: 1.1rem;
            }

            .hero-card h1 {
                margin: 0 0 0.45rem 0;
                color: #5b44a5;
                font-size: 2.55rem;
                line-height: 1.1;
            }

            .hero-card p {
                color: #514a6b;
                font-size: 1.09rem;
                margin-bottom: 0;
            }

            .friendly-card {
                padding: 1rem 1.15rem;
                border-radius: 24px;
                background: rgba(255, 255, 255, 0.94);
                border: 1px solid rgba(172, 205, 255, 0.68);
                box-shadow: 0 10px 25px rgba(47, 76, 128, 0.07);
                margin-bottom: 0.9rem;
            }

            .tip-card {
                padding: 0.95rem 1.1rem;
                border-radius: 22px;
                background: #fff8dd;
                border: 1px solid #ffe08a;
                color: #56451f;
                margin-bottom: 0.85rem;
            }

            .caption-card {
                padding: 1rem 1.2rem;
                border-radius: 22px;
                background: #f1fbff;
                border-left: 8px solid #7ac7ff;
                color: #334155;
                font-size: 1rem;
                margin-bottom: 1rem;
            }

            .story-card {
                padding: 1.35rem 1.5rem;
                border-radius: 28px;
                background: #fffef8;
                border: 2px dashed #ffc857;
                line-height: 1.9;
                font-size: 1.12rem;
                color: #3d3a4b;
                box-shadow: 0 12px 28px rgba(255, 200, 87, 0.18);
                margin-bottom: 0.8rem;
            }

            .step-done, .step-active, .step-pending {
                padding: 0.78rem 1rem;
                border-radius: 18px;
                margin: 0.45rem 0;
                font-size: 0.98rem;
            }

            .step-done {
                background: #ecfff3;
                border: 1px solid #aee8bd;
            }

            .step-active {
                background: #fff7da;
                border: 1px solid #f0d06e;
            }

            .step-pending {
                background: #ffffff;
                border: 1px solid #e8e8e8;
                color: #777;
            }

            .choice-chip {
                display: inline-block;
                padding: 0.45rem 0.7rem;
                border-radius: 999px;
                background: #f7edff;
                border: 1px solid #dfc6ff;
                color: #59428c;
                font-weight: 700;
                margin: 0.15rem 0.25rem 0.15rem 0;
            }

            .success-banner {
                padding: 1rem 1.15rem;
                border-radius: 24px;
                background: linear-gradient(135deg, #edfff4 0%, #f2fbff 100%);
                border: 1px solid #bdeecb;
                color: #275339;
                margin-bottom: 0.9rem;
            }

            section[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #fff4fb 0%, #eef8ff 100%);
            }

            section[data-testid="stSidebar"] h2,
            section[data-testid="stSidebar"] h3 {
                color: #5b44a5;
            }

            .stButton > button, .stDownloadButton > button {
                border-radius: 999px;
                font-weight: 800;
                min-height: 3rem;
                border: 0;
                box-shadow: 0 8px 18px rgba(116, 92, 199, 0.13);
            }

            .stFileUploader {
                padding: 0.2rem 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render the main app title and short child-friendly description."""
    st.markdown(
        """
        <div class="hero-card">
            <h1>🌈 Little Dream Storyteller</h1>
            <p>
                Pick a picture, choose a story mood, and make a tiny storybook with a gentle reading voice.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_story_controls() -> tuple[str, str, bool]:
    """Collect simple story options without showing technical settings."""
    st.sidebar.markdown("## 🎨 Make Your Story")

    child_name = st.sidebar.text_input(
        "Hero name",
        placeholder="e.g. Emma, Leo, Lily",
        help="Optional. The story can use this name as the main character.",
    )

    story_style_label = st.sidebar.radio(
        "Choose a story mood",
        list(STORY_STYLES.keys()),
        index=0,
        help="This changes the feeling of the story.",
    )

    reading_speed_label = st.sidebar.radio(
        "Reading voice",
        list(READING_SPEEDS.keys()),
        index=0,
        help="Slow reading is better for young children and bedtime stories.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div class="friendly-card">
            <b>How to play</b><br><br>
            1. Upload a picture 🖼️<br>
            2. Pick a story mood ✨<br>
            3. Press the big story button 📖<br>
            4. Listen to the story 🔊
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        """
        <div class="tip-card">
            💡 Try a clear picture with animals, toys, people, food, or outdoor scenes.
        </div>
        """,
        unsafe_allow_html=True,
    )

    return child_name, story_style_label, READING_SPEEDS[reading_speed_label]


def render_progress(stage: str) -> None:
    """Show a simple three-step progress panel."""
    states_by_stage = {
        "idle": ["pending", "pending", "pending"],
        "captioning": ["active", "pending", "pending"],
        "story": ["done", "active", "pending"],
        "audio": ["done", "done", "active"],
        "done": ["done", "done", "done"],
    }

    labels = [
        "Look at the picture",
        "Write a tiny story",
        "Read it aloud",
    ]

    icons = {
        "done": "✅ Ready",
        "active": "🌟 Making magic",
        "pending": "⬜ Waiting",
    }

    css_classes = {
        "done": "step-done",
        "active": "step-active",
        "pending": "step-pending",
    }

    states = states_by_stage.get(stage, states_by_stage["idle"])

    for label, state in zip(labels, states):
        st.markdown(
            f"<div class='{css_classes[state]}'><b>{label}</b>&nbsp;&nbsp;{icons[state]}</div>",
            unsafe_allow_html=True,
        )


# =============================================================================
# 3. Text processing and validation
# =============================================================================

def normalize_text(text: str) -> str:
    """Remove extra spaces and fix simple punctuation spacing."""
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text


def count_words(text: str) -> int:
    """Count English-like words in a text string."""
    return len(re.findall(r"\b[\w'-]+\b", text or ""))


def create_image_hash(image_bytes: bytes) -> str:
    """Create a stable image identifier for session-state management."""
    return hashlib.md5(image_bytes).hexdigest()


def clean_caption(caption: str) -> str:
    """Clean the raw image caption generated by the image captioning model."""
    caption = normalize_text(caption)
    caption = re.sub(
        r"^(a picture of|an image of|there is|there are)\s+",
        "",
        caption,
        flags=re.IGNORECASE,
    )
    caption = caption.strip(" .,;:")

    if count_words(caption) < 3:
        return "a friendly child discovering something interesting in a bright place"

    return caption


def split_into_sentences(text: str) -> list[str]:
    """Split story text into simple sentences."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text or "") if s.strip()]


def clean_story_text(text: str) -> str:
    """Clean model output so the app displays only a child-friendly story."""
    story = normalize_text(text)
    story = story.strip().strip('"').strip("'")
    story = re.sub(r"^(Story:|Title:)", "", story, flags=re.IGNORECASE).strip()

    unwanted_markers = [
        "Explanation:",
        "Requirements:",
        "Prompt:",
        "Caption:",
        "Image:",
        "Here is",
    ]

    for marker in unwanted_markers:
        marker_index = story.find(marker)
        if marker_index > 15:
            story = story[:marker_index].strip()

    sentences = split_into_sentences(story)
    kept_sentences: list[str] = []

    for sentence in sentences:
        candidate = " ".join(kept_sentences + [sentence])
        if count_words(candidate) <= MAX_STORY_WORDS:
            kept_sentences.append(sentence)

    story = " ".join(kept_sentences).strip() if kept_sentences else story

    if count_words(story) > MAX_STORY_WORDS:
        story = " ".join(story.split()[:MAX_STORY_WORDS]).rstrip(" ,;:") + "."

    if story and story[-1] not in ".!?":
        story += "."

    return story


def build_fallback_story(caption: str, story_style_label: str, child_name: str) -> str:
    """Create a safe fallback story if the language model output is too short or messy."""
    hero = child_name.strip() if child_name.strip() else "the little child"
    style_key = STORY_STYLES[story_style_label]["style_key"]
    style_word = {
        "Magic": "sparkly",
        "Adventure": "brave",
        "Friendship": "kind",
        "Bedtime": "gentle",
    }.get(style_key, "happy")

    return (
        f"One bright day, {hero} saw {caption}. It looked so {style_word} that {hero} wanted to learn more. "
        f"Suddenly, a tiny problem appeared: something important seemed to be missing. {hero} looked around carefully and asked for help kindly. "
        f"Together, everyone found the missing piece and shared a big smile. By the end of the day, {hero} felt proud, safe, and very happy."
    )


def enforce_story_length(story: str, caption: str, story_style_label: str, child_name: str) -> str:
    """Ensure the final story is close to the required 50-100 word range."""
    story = clean_story_text(story)
    word_count = count_words(story)

    if MIN_STORY_WORDS <= word_count <= MAX_STORY_WORDS:
        return story

    fallback = build_fallback_story(caption, story_style_label, child_name)
    fallback = clean_story_text(fallback)

    if count_words(fallback) > MAX_STORY_WORDS:
        fallback = " ".join(fallback.split()[:MAX_STORY_WORDS]).rstrip(" ,;:") + "."

    return fallback


def prepare_story_prompt(caption: str, story_style_label: str, child_name: str) -> str:
    """Create a compact prompt suitable for DistilGPT-2 text generation."""
    hero = child_name.strip() if child_name.strip() else "a little child"
    style_description = STORY_STYLES[story_style_label]["description"]

    return (
        "Write a simple, safe children's story for ages 3 to 10. "
        f"The story is about {hero} and {caption}. "
        f"The mood is {style_description}. "
        "The story has a beginning, a small problem, a kind solution, and a happy ending. Story:"
    )


def remove_prompt_from_generated_text(generated_text: str, prompt: str) -> str:
    """Remove the prompt prefix from a text-generation model output."""
    if generated_text.startswith(prompt):
        return generated_text[len(prompt):].strip()
    return generated_text.strip()


def prepare_text_for_child_narration(story: str) -> str:
    """Improve gTTS pacing by adding clear sentence breaks."""
    sentences = split_into_sentences(story)
    if not sentences:
        return story

    polished_sentences: list[str] = []
    for sentence in sentences:
        sentence = normalize_text(sentence)
        if not sentence:
            continue
        if sentence[-1] not in ".!?":
            sentence += "."
        polished_sentences.append(sentence)

    return "\n\n".join(polished_sentences)


# =============================================================================
# 4. Model loading
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_captioner() -> Any:
    """Load the image captioning pipeline."""
    return pipeline("image-to-text", model=IMAGE_CAPTION_MODEL)


@st.cache_resource(show_spinner=False)
def load_story_generator() -> Any:
    """Load the story generation pipeline."""
    return pipeline("text-generation", model=STORY_GENERATION_MODEL)


# =============================================================================
# 5. Core app logic
# =============================================================================

def generate_caption(image: Image.Image) -> str:
    """Generate a caption from the uploaded image."""
    captioner = load_captioner()
    result = captioner(image)

    if isinstance(result, list) and result:
        raw_caption = result[0].get("generated_text", "")
    else:
        raw_caption = ""

    return clean_caption(raw_caption)


def generate_story(caption: str, story_style_label: str, child_name: str) -> str:
    """Generate a 50-100 word story from the caption using DistilGPT-2."""
    generator = load_story_generator()
    prompt = prepare_story_prompt(caption, story_style_label, child_name)

    output = generator(
        prompt,
        max_new_tokens=90,
        do_sample=True,
        temperature=0.85,
        top_p=0.92,
        repetition_penalty=1.18,
        no_repeat_ngram_size=3,
        pad_token_id=generator.tokenizer.eos_token_id,
        num_return_sequences=1,
    )

    generated_text = output[0].get("generated_text", "") if output else ""
    raw_story = remove_prompt_from_generated_text(generated_text, prompt)

    return enforce_story_length(raw_story, caption, story_style_label, child_name)


def text_to_speech(story: str, slow_reading: bool) -> tuple[Optional[bytes], str]:
    """Convert story text to child-friendly speech using gTTS."""
    try:
        narration_text = prepare_text_for_child_narration(story)
        audio_buffer = BytesIO()

        tts = gTTS(
            text=narration_text,
            lang="en",
            slow=slow_reading,
        )
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)

        voice_label = "Gentle slow reading" if slow_reading else "Clear reading"
        return audio_buffer.read(), voice_label

    except Exception as error:
        st.warning(f"Audio generation failed: {error}")
        return None, "Audio generation failed"


# =============================================================================
# 6. Result rendering
# =============================================================================

def render_empty_storybook() -> None:
    """Render the initial empty state."""
    st.markdown(
        """
        <div class="friendly-card">
            <h3 style="margin-top:0; color:#5b44a5;">📚 Your storybook is waiting</h3>
            <p>Upload a picture on the left, then press the big story button.</p>
            <span class="choice-chip">Picture</span>
            <span class="choice-chip">Story</span>
            <span class="choice-chip">Voice</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_panel(result: dict[str, Any]) -> None:
    """Render caption, story, word count, audio, and download buttons."""
    status = result.get("status", "idle")

    if status == "idle":
        render_empty_storybook()
    elif status == "captioning":
        st.markdown("<div class='friendly-card'>🖼️ Looking carefully at the picture...</div>", unsafe_allow_html=True)
    elif status == "story":
        st.markdown("<div class='friendly-card'>📖 Writing a tiny story...</div>", unsafe_allow_html=True)
    elif status == "audio":
        st.markdown("<div class='friendly-card'>🔊 Preparing a gentle reading voice...</div>", unsafe_allow_html=True)
    elif status == "done":
        st.markdown("<div class='success-banner'>🌟 Hooray! Your storybook is ready.</div>", unsafe_allow_html=True)

    render_progress(status)

    caption = result.get("caption")
    story = result.get("story")
    audio = result.get("audio")
    voice_label = result.get("voice_label")
    result_id = result.get("result_id", "current")

    if caption:
        st.subheader("🖼️ I noticed...")
        st.markdown(f"<div class='caption-card'>{caption}</div>", unsafe_allow_html=True)

    if story:
        st.subheader("📖 Your tiny story")
        st.markdown(f"<div class='story-card'>{story}</div>", unsafe_allow_html=True)

        word_count = count_words(story)
        st.caption(f"Story length: {word_count} words")

        if not (MIN_STORY_WORDS <= word_count <= MAX_STORY_WORDS):
            st.warning("This story is outside the target length. Please create another one.")

        if status == "done":
            st.download_button(
                "💾 Save story text",
                data=story,
                file_name="little_dream_story.txt",
                mime="text/plain",
                use_container_width=True,
                key=f"download_story_{result_id}",
            )

    if audio:
        st.subheader("🔊 Listen to the story")
        if voice_label:
            st.caption(voice_label)
        st.audio(audio, format="audio/mp3")
        if status == "done":
            st.download_button(
                "🎧 Save story audio",
                data=audio,
                file_name="little_dream_story_audio.mp3",
                mime="audio/mpeg",
                use_container_width=True,
                key=f"download_audio_{result_id}",
            )


# =============================================================================
# 7. Main Streamlit app
# =============================================================================

def main() -> None:
    """Run the Streamlit storytelling application."""
    inject_custom_css()
    render_header()

    child_name, story_style_label, slow_reading = render_story_controls()

    if "result" not in st.session_state:
        st.session_state.result = {"status": "idle"}

    left_column, right_column = st.columns([1.03, 0.97], gap="large")

    with left_column:
        st.subheader("🖼️ Step 1: Choose a picture")
        st.markdown(
            """
            <div class="friendly-card">
                Pick a clear image. Animals, toys, family moments, food, or nature pictures work especially well.
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Drop a picture here",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        image: Optional[Image.Image] = None
        current_image_hash: Optional[str] = None

        if uploaded_file is not None:
            image_bytes = uploaded_file.getvalue()
            current_image_hash = create_image_hash(image_bytes)
            image = Image.open(BytesIO(image_bytes)).convert("RGB")

            st.image(image, caption="Your picture", use_container_width=True)

            if st.session_state.result.get("image_hash") not in (None, current_image_hash):
                st.session_state.result = {
                    "status": "idle",
                    "image_hash": current_image_hash,
                }

            st.markdown("### 📖 Step 2: Make the story")
            selected_style_hint = STORY_STYLES[story_style_label]["button_hint"]
            st.markdown(
                f"<div class='tip-card'>{selected_style_hint}</div>",
                unsafe_allow_html=True,
            )

            create_button_clicked = st.button(
                "✨ Create my storybook",
                type="primary",
                use_container_width=True,
            )
        else:
            create_button_clicked = False
            st.markdown(
                """
                <div class="tip-card">
                    🌟 Start by uploading a JPG or PNG picture.
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right_column:
        st.subheader("📚 Storybook")
        result_placeholder = st.empty()

        def refresh_result_panel() -> None:
            """Clear and redraw the right result panel."""
            result_placeholder.empty()
            with result_placeholder.container():
                render_result_panel(st.session_state.result)

        refresh_result_panel()

    if create_button_clicked and image is not None and current_image_hash is not None:
        try:
            run_id = st.session_state.get("run_id", 0) + 1
            st.session_state.run_id = run_id
            result_id = f"{current_image_hash}_{run_id}"

            st.session_state.result = {
                "status": "captioning",
                "image_hash": current_image_hash,
                "result_id": result_id,
            }
            refresh_result_panel()

            caption = generate_caption(image)
            st.session_state.result.update({"caption": caption, "status": "story"})
            refresh_result_panel()

            story = generate_story(caption, story_style_label, child_name)
            st.session_state.result.update({"story": story, "status": "audio"})
            refresh_result_panel()

            audio, voice_label = text_to_speech(story, slow_reading)
            st.session_state.result.update(
                {
                    "audio": audio,
                    "voice_label": voice_label,
                    "status": "done",
                }
            )
            refresh_result_panel()

        except Exception as error:
            st.error("Something went wrong while making the storybook.")
            st.exception(error)

    st.markdown("---")
    st.caption("Made for young readers with a simple picture-to-story-to-audio experience.")


if __name__ == "__main__":
    main()
