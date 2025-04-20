# streamlit_app.py

import io
import json
import requests
import nltk
from nltk.tokenize import word_tokenize
import streamlit as st
import speech_recognition as sr
from pydub import AudioSegment
from PIL import Image
from gtts import gTTS
from langchain.llms import OpenAI
from langchain.agents import AgentType, initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import JinaChat
from langchain.tools import DuckDuckGoSearchRun

# — Download tokenizer once
nltk.download("punkt")

# ======== CONFIGURATION – insert your keys here ========
HUGGINGFACE_API_KEY = "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"
JINACHAT_API_KEY    = "CTh7GuLXNG6O7SZ8mq9y:121102ad7772ce8418d7c6818e59af102c2131c6e5fcdb01c9052155e21a132c"
hf_headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

# ======== HELPERS ========
def split_into_meaningful_words(text: str) -> str:
    words = word_tokenize(text)
    return ", ".join(w for w in words if w.isalnum())

def text_summarization_query(text: str) -> dict:
    API_URL = "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6"
    payload = {
        "inputs": text + " -- Please summarize into actionable keywords (max 20 words).",
        "options": {"wait_for_model": True},
    }
    r = requests.post(API_URL, headers=hf_headers, json=payload)
    r.raise_for_status()
    return r.json()

def text_to_image_query(prompt: str) -> bytes:
    API_URL = "https://api-inference.huggingface.co/models/artificialguybr/IconsRedmond-IconsLoraForSDXL"
    payload = {"inputs": prompt, "options": {"wait_for_model": True}}
    r = requests.post(API_URL, headers=hf_headers, json=payload)
    r.raise_for_status()
    return r.content

# ======== STREAMLIT UI ========
st.set_page_config(page_title="DiagnoAI", page_icon="🤖", layout="centered")
st.title("🤖 DiagnoAI : Health first!")

recognizer = sr.Recognizer()

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Audio uploader
audio_file = st.file_uploader("Upload your audio file (WAV/MP3/etc.)", type=["wav","mp3","m4a","flac","ogg"])
if audio_file:
    # Read raw bytes & show player
    raw = audio_file.read()
    st.audio(raw, format=f"audio/{audio_file.type.split('/')[-1]}")
    
    # Convert *any* format to pure PCM‑WAV via pydub+ffmpeg
    audio_buffer = io.BytesIO(raw)
    sound = AudioSegment.from_file(audio_buffer)
    pcm_wav = io.BytesIO()
    sound.export(pcm_wav, format="wav")   # PCM by default
    pcm_wav.seek(0)

    # Transcribe
    with sr.AudioFile(pcm_wav) as src:
        audio_data = recognizer.record(src)
    try:
        text = recognizer.recognize_google(audio_data)
    except sr.RequestError:
        st.error("Could not reach Google Speech API.")
        st.stop()
    except sr.UnknownValueError:
        st.error("Audio not clear enough to transcribe.")
        st.stop()

    # Append & display user text
    st.session_state.messages.append({"role": "user", "content": text})
    st.chat_message("user").write(text)

    # Initialize JinaChat agent
    chat = JinaChat(temperature=0.2, streaming=True, jinachat_api_key=JINACHAT_API_KEY)
    agent = initialize_agent(
        tools=[DuckDuckGoSearchRun(name="Search")],
        llm=chat,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=False,
    )

    # Get AI response
    with st.chat_message("assistant"):
        cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        reply = agent.run(st.session_state.messages, callbacks=[cb])
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.write(reply)

    # Text-to-speech
    out_fp = "output.wav"
    gTTS(text=reply, lang="en", slow=False).save(out_fp)
    st.audio(out_fp, format="audio/wav")

    # Summarize & generate image
    with st.spinner("Generating image..."):
        summ = text_summarization_query(reply)
        summ_txt = summ[0].get("summary_text", "") if isinstance(summ, list) else str(summ)
        kws = split_into_meaningful_words(summ_txt)
        prompt = f"{kws}, 1 human, english language, exercise, healthy diet, medicines, vegetables, fruits"
        img_bytes = text_to_image_query(prompt)
        st.image(Image.open(io.BytesIO(img_bytes)), use_column_width=True)
