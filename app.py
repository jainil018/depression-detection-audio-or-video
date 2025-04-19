from flask import Flask, request, render_template, jsonify
import os
import subprocess
import joblib
import pandas as pd
import whisper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import uuid
from datetime import datetime
from openpyxl import load_workbook

app = Flask(__name__, template_folder='templates')

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Allowed audio/video extensions
ALLOWED_EXTENSIONS = {'mp3', 'mp4', 'wav', 'm4a', 'aac', 'webm'}

# Load Whisper model
whisper_model = whisper.load_model("base")  # Use "small", "medium", or "large" for better accuracy

# Load and train the model from Excel
def train_model():
    df = pd.read_excel("processed_transcriptions_data.xlsx")
    X = df['transcription'].astype(str)
    y = df['label']

    vectorizer = TfidfVectorizer()
    X_features = vectorizer.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_features, y)

    joblib.dump(model, "depression_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

    return model, vectorizer, df

# Load trained model and vectorizer
model, tfidf_vectorizer, df_train = train_model()

# File extension validation
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload an audio or video file.'}), 400

    # Generate unique filename
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Transcribe audio to text
    text = transcribe_audio(file_path)
    if not text:
        return jsonify({'error': 'Could not transcribe audio'}), 500

    # Predict depression
    text_features = tfidf_vectorizer.transform([text])
    prediction = model.predict(text_features)[0]

    result = "Depression detected" if prediction == 1 else "No depression detected"
    explanation = analyze_text(text, prediction)

    # Save to Excel
    save_user_result(filename, text, prediction, explanation)

    return jsonify({'transcription': text, 'result': result, 'explanation': explanation})

def transcribe_audio(video_path):
    try:
        # Generate unique temp audio file
        audio_path = f"temp_audio_{uuid.uuid4().hex}.wav"
        command = f'ffmpeg -i "{video_path}" -ac 1 -ar 16000 -acodec pcm_s16le "{audio_path}" -y'
        subprocess.run(command, shell=True, check=True)

        # Transcribe with Whisper
        transcription = whisper_model.transcribe(audio_path)['text']

        os.remove(audio_path)
        return transcription.strip()

    except Exception as e:
        print("Error in transcription:", e)
        return None

def analyze_text(input_text, prediction):
    if prediction == 1:
        return "Patterns in your speech match depressive speech data."

    non_depressive_texts = df_train[df_train['label'] == 0]['transcription']
    common_words = set(" ".join(non_depressive_texts).split()) & set(input_text.split())

    return f"Your speech contains neutral words like: {', '.join(list(common_words)[:5])}."

def save_user_result(filename, transcription, prediction, explanation):
    output_file = "user_results.xlsx"
    new_data = {
        "Timestamp": [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
        "File Name": [filename],
        "Transcription": [transcription],
        "Prediction": [prediction],
        "Explanation": [explanation]
    }

    new_df = pd.DataFrame(new_data)

    # Append to existing Excel file or create a new one
    if os.path.exists(output_file):
        with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            existing = pd.read_excel(output_file)
            start_row = existing.shape[0] + 1
            new_df.to_excel(writer, index=False, header=False, startrow=start_row)
    else:
        new_df.to_excel(output_file, index=False)

if __name__ == '__main__':
    app.run(debug=True)
