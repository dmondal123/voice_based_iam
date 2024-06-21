from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import Optional
import numpy as np
import tensorflow as tf
import joblib
import librosa
import os
from glob import glob
from pydub import AudioSegment
import torch
import torchaudio
from audioseal import AudioSeal
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:5173",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# Load the model
model = tf.keras.models.load_model('speaker_recognition_model.h5')

# Load the label encoder and scaler
le = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')

# Function to extract features
def extract_features(file_name):
    try:
        audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}. Error: {e}")
        return None
    return mfccs_scaled

# Function to predict speaker with confidence threshold
def predict_speaker(file_path, threshold=0.75):
    features = extract_features(file_path)
    if features is not None:
        features = scaler.transform([features])
        predictions = model.predict(features)
        prediction_confidence = np.max(predictions)
        if prediction_confidence >= threshold:
            prediction = np.argmax(predictions, axis=1)
            speaker = le.inverse_transform(prediction)
            return speaker[0]
        else:
            return "other"
    else:
        return None


def convert_to_wav(audio_path):
    if not audio_path.endswith('.wav'):
        new_path = audio_path.rsplit('.', 1)[0] + '.wav'
        audio = AudioSegment.from_file(audio_path)
        audio.export(new_path, format='wav')
        return new_path
    return audio_path

def load_audio(file_path, target_sample_rate=16000):
    file_path=convert_to_wav(file_path)
    
    wav,sr = torchaudio.load(file_path)
    wav,sr=torchaudio.load(file_path)

    if sr != target_sample_rate:
        wav = torchaudio.transforms.Resample(sr, target_sample_rate)(wav)
        sr = target_sample_rate
    return wav, sr

def detect_watermark(audio_path):
    # Load the audio file 
    wav, sr = load_audio(audio_path)
    
    # Ensure the audio tensor has the right shape (batch, channels, samples)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)  # add batch dimension if not present
    if wav.dim() == 2:
        wav = wav.unsqueeze(0)  # add channel dimension if not present

    # Load the detector model
    detector = AudioSeal.load_detector("audioseal_detector_16bits")
    
    # Detect watermark
    result, message = detector.detect_watermark(wav, sr)

    # Determine if the audio is likely cloned
    if result > 0.5:
        return True, "Voice clone detected. Authorization Failure"
    else:
        return False, "The audio is unlikely to be watermarked (cloned)."
    
def is_silent(audio_data, threshold=0.01):
    # Calculate the root mean square (RMS) energy of the audio data
    rms = librosa.feature.rms(y=audio_data).mean()
    return rms < threshold


# Define FastAPI endpoint for speaker prediction
@app.post("/predict/")
async def predict(file: UploadFile = File(...), threshold: Optional[float] = 0.75):
    # Save the uploaded file temporarily
    with open(file.filename, "wb") as buffer:
        buffer.write(file.file.read())
        file_path = file.filename
    
    audio_data, _ = librosa.load(file_path, sr=None)
    if is_silent(audio_data):
        predicted_speaker="No Voice"
        return {"predicted_speaker": predicted_speaker}
    
    is_cloned, message = detect_watermark(file_path)
    
    if is_cloned:
        predicted_speaker = "Cloned"
        os.remove(file_path)  # Clean up the saved file
        return {"predicted_speaker": predicted_speaker}


    # Perform speaker prediction
    predicted_speaker = predict_speaker(file_path, threshold)
    if predicted_speaker:
        return {"predicted_speaker": predicted_speaker}
    else:
        raise HTTPException(status_code=404, detail=f"Could not predict speaker for {file.filename}")

# Run the FastAPI application with Uvicorn server
if __name__ == "__main__":
    import uvicorn
    print("hello ")
    uvicorn.run(app, host="0.0.0.0", port=3000)





