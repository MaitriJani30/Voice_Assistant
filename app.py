import io
import time
from google import genai
from google.genai import types
from gtts import gTTS
import speech_recognition as sr
import streamlit as st

st.set_page_config(page_title="AI Voice Assistant", page_icon="🎙️")
st.title("🎙️ Gemini AI Voice Assistant")

# Sidebar setup (loads saved key from .streamlit/secrets.toml if available, otherwise asks in sidebar)
with st.sidebar:
  st.header("Settings")
  if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
    st.success("API Key loaded from secrets!")
  else:
    api_key = st.text_input("Enter your Gemini API Key:", type="password")

  if st.button("Clear Conversation"):
    st.session_state.messages = []
    st.session_state.last_audio_bytes = None
    st.rerun()

# Initialize session memory
if "messages" not in st.session_state:
  st.session_state.messages = []
if "last_audio_bytes" not in st.session_state:
  st.session_state.last_audio_bytes = None


# Convert text reply into browser-playable MP3 audio
def text_to_speech(text):
  clean_text = text.replace("*", "").replace("#", "")
  tts = gTTS(text=clean_text, lang="en")
  audio_fp = io.BytesIO()
  tts.write_to_fp(audio_fp)
  audio_fp.seek(0)
  return audio_fp


# Send conversation to Gemini with automatic fallback if a model has high demand
def generate_ai_response(api_key):
  client = genai.Client(api_key=api_key)
  formatted_history = []
  for msg in st.session_state.messages:
    role = "user" if msg["role"] == "user" else "model"
    formatted_history.append(
        types.Content(role=role, parts=[types.Part.from_text(text=msg["content"])])
    )

  models_to_try = [
      "gemini-3.5-flash-lite",
      "gemini-3.1-flash-lite",
      "gemini-3.8-flash",
  ]

  last_error = None
  for model_name in models_to_try:
    try:
      response = client.models.generate_content(
          model=model_name,
          contents=formatted_history,
          config=types.GenerateContentConfig(
              system_instruction=(
                  "You are a helpful voice assistant. Keep your answers"
                  " concise, conversational, and under 3 sentences so they"
                  " sound natural when spoken aloud."
              )
          ),
      )
      return response.text
    except Exception as e:
      last_error = e
      err_msg = str(e).lower()
      if (
          "503" in err_msg
          or "unavailable" in err_msg
          or "high demand" in err_msg
          or "429" in err_msg
      ):
        time.sleep(1)
        continue
      else:
        raise e

  raise RuntimeError(
      f"All models are currently busy. Please try again. ({last_error})"
  )


# 1. Chat history container at the TOP
chat_container = st.container()
with chat_container:
  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])

# 2. Microphone recorder at the BOTTOM
audio_value = st.audio_input("Click the microphone, speak, then click Stop")

if audio_value is not None:
  current_bytes = audio_value.getvalue()

  # Process only if it is a new recording
  if current_bytes != st.session_state.last_audio_bytes:
    if not api_key:
      st.warning("Please enter your Gemini API Key in the left sidebar first.")
    else:
      st.session_state.last_audio_bytes = current_bytes
      recognizer = sr.Recognizer()

      try:
        with st.spinner("Transcribing..."):
          with sr.AudioFile(io.BytesIO(current_bytes)) as source:
            audio_data = recognizer.record(source)
          user_command = recognizer.recognize_google(audio_data)

        # Save and display user message
        st.session_state.messages.append(
            {"role": "user", "content": user_command}
        )
        with chat_container:
          with st.chat_message("user"):
            st.markdown(user_command)

        # Generate, save, display, and speak AI response
        with st.spinner("Thinking..."):
          bot_reply = generate_ai_response(api_key)

        st.session_state.messages.append(
            {"role": "assistant", "content": bot_reply}
        )
        with chat_container:
          with st.chat_message("assistant"):
            st.markdown(bot_reply)
            audio_bytes = text_to_speech(bot_reply)
            st.audio(audio_bytes, format="audio/mp3", autoplay=True)

      except sr.UnknownValueError:
        st.error(
            "Could not understand the audio. Please speak clearly and try"
            " again."
        )
      except Exception as e:
        st.error(f"Error: {e}")
