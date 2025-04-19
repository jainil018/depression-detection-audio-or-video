import gradio as gr
import whisper
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import os
import uuid
import subprocess

# Load Whisper model (tiny to fit on HF Spaces)
whisper_model = whisper.load_model("tiny")

# Train model from Excel
def train_model():
    df = pd.read_excel("processed_transcriptions_data.xlsx")
    X = df['transcription'].astype(str)
    y = df['label']
    vectorizer = TfidfVectorizer()
    X_features = vectorizer.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_features, y)
    return model, vectorizer, df

model, vectorizer, df_train = train_model()

# Transcription function
def transcribe_audio(file_path):
    try:
        temp_audio = f"temp_{uuid.uuid4().hex}.wav"
        subprocess.run(
            ["ffmpeg", "-i", file_path, "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le", temp_audio],
            check=True
        )
        result = whisper_model.transcribe(temp_audio)
        os.remove(temp_audio)
        return result["text"]
    except Exception as e:
        print("Error:", e)
        return None

# Explain why it's not depression
def analyze_text(input_text, prediction):
    if prediction == 1:
        return "Patterns in your speech match depressive speech data."
    else:
        non_depressive_texts = df_train[df_train['label'] == 0]['transcription']
        common_words = set(" ".join(non_depressive_texts).split()) & set(input_text.split())
        return f"Your speech contains neutral words like: {', '.join(list(common_words)[:5])}."

# Gradio inference function
def predict_depression(file):
    transcription = transcribe_audio(file.name)
    if not transcription:
        return "Transcription failed.", ""
    features = vectorizer.transform([transcription])
    prediction = model.predict(features)[0]
    result = "Depression detected" if prediction == 1 else "No depression detected"
    explanation = analyze_text(transcription, prediction)
    return result, explanation

# Gradio interface
interface = gr.Interface(
    fn=predict_depression,
    inputs=gr.Audio(type="filepath", label="Upload Audio or Video"),
    outputs=["text", "text"],
    title="Depression Detection from Audio/Video",
    description="Upload an audio or video file to detect signs of depression using Whisper and ML."
)

interface.launch()
