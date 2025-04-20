# streamlit_app.py

import io
import json
import requests
import nltk
from nltk.tokenize import word_tokenize
import streamlit as st
import speech_recognition as sr
from PIL import Image
from gtts import gTTS
from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import JinaChat
from langchain.tools import DuckDuckGoSearchRun

# Download NLTK data once
nltk.download("punkt")

# ============ Configuration – fill in your keys ============
HUGGINGFACE_API_KEY = "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"
JINACHAT_API_KEY    = "CTh7GuLXNG6O7SZ8mq9y:121102ad7772ce8418d7c6818e59af102c2131c6e5fcdb01c9052155e21a132c"
headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

# ============ Helper functions ============

def split_into_meaningful_words(text: str) -> str:
    words = word_tokenize(text)
    meaningful = [w for w in words if w.isalnum()]
    return ", ".join(meaningful)


def text_summarization_query(text: str) -> dict:
    API_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"
    payload = {
        "inputs": text + " -- Please summarize into actionable keywords (max 20 words).",
        "options": {"wait_for_model": True},
    }
    resp = requests.post(API_URL, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()


def text_to_image_query(prompt: str) -> bytes:
    API_URL = "https://api-inference.huggingface.co/models/artificialguybr/IconsRedmond-IconsLoraForSDXL"
    payload = {"inputs": prompt, "options": {"wait_for_model": True}}
    resp = requests.post(API_URL, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.content


# ============ Streamlit setup ============

st.set_page_config(page_title="DiagnoAI", page_icon="🤖", layout="centered")
st.title("🤖 DiagnoAI : Health first!")

# Initialize recognizer
recognizer = sr.Recognizer()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you?"},
    ]

# Display chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# File uploader
audio_file = st.file_uploader("Upload your audio file (WAV)", type="wav")
if audio_file is not None:
    st.audio(audio_file)

    # —— Use the UploadedFile directly as a file-like object —— #
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)

    # Transcribe
    try:
        transcribed_text = recognizer.recognize_google(audio_data)
    except sr.RequestError:
        st.error("Could not reach Google Speech API.")
        st.stop()
    except sr.UnknownValueError:
        st.error("Audio was not clear enough to transcribe.")
        st.stop()

    # Append & display user message
    st.session_state.messages.append({"role": "user", "content": transcribed_text})
    st.chat_message("user").write(transcribed_text)

    # Initialize JinaChat
    chat = JinaChat(
        temperature=0.2,
        streaming=True,
        jinachat_api_key=JINACHAT_API_KEY,
    )

    # Initialize the agent
    agent = initialize_agent(
        tools=[DuckDuckGoSearchRun(name="Search")],
        llm=chat,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=False,
    )

    # Get AI response
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response_text = agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response_text})
        st.write(response_text)

    # Text‑to‑speech
    out_file = "output.wav"
    tts = gTTS(text=response_text, lang="en", slow=False)
    tts.save(out_file)
    st.audio(out_file)

    # Summarize → generate image
    with st.spinner("Generating image..."):
        summary = text_summarization_query(response_text)
        # summary is usually a list of dicts, adjust if needed:
        summary_text = summary[0]["summary_text"] if isinstance(summary, list) else str(summary)
        keywords = split_into_meaningful_words(summary_text)
        prompt = f"{keywords}, 1 human, english language, exercise, healthy diet, medicines, vegetables, fruits"
        img_bytes = text_to_image_query(prompt)
        img = Image.open(io.BytesIO(img_bytes))
        st.image(img, use_column_width=True)
