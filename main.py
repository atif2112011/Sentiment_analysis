from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import librosa
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load the model
model_path = "Emotion_Detection_1.keras"
model = load_model(model_path)

# Class names corresponding to the 8 emotion classes
class_names = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

# Define a root endpoint to check the server status
@app.get("/check")
def check():
    return {"message": "Server is running"}

# Function to process the audio and extract MFCC features
def process_audio(file_path, segment_length=4.0, sr=22050, n_mfcc=13, n_fft=2048, hop_length=512):
    audio, sr = librosa.load(file_path, sr=sr)
    samples_per_segment = int(segment_length * sr)
    expected_frames_per_segment = 180
    mfcc_segments = []

    for i in range(0, len(audio), samples_per_segment):
        segment = audio[i:i + samples_per_segment]

        if len(segment) < samples_per_segment:
            segment = np.pad(segment, (0, samples_per_segment - len(segment)), mode='constant')

        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        if mfcc.shape[1] < expected_frames_per_segment:
            mfcc = np.pad(mfcc, ((0, 0), (0, expected_frames_per_segment - mfcc.shape[1])), mode='constant')

        mfcc_segments.append(mfcc)

    return mfcc_segments

# Function to predict emotion based on MFCC segments
def predict_and_average(mfcc_segments, model, class_names):
    total_prediction = np.zeros((len(class_names),))
    
    for mfcc in mfcc_segments:
        mfcc_input = np.expand_dims(mfcc, axis=0)
        prediction = model.predict(mfcc_input)
        total_prediction += prediction[0]

    avg_prediction = total_prediction / len(mfcc_segments)
    predicted_class_idx = np.argmax(avg_prediction)
    predicted_class_name = class_names[predicted_class_idx]

    return predicted_class_name, avg_prediction

# Define the /predict endpoint to handle audio file uploads
@app.post("/predict")
async def predict_emotion(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    file_location = f"/tmp/{file.filename}"
    
    with open(file_location, "wb") as f:
        f.write(await file.read())

    # Process audio and extract MFCC features
    mfcc_segments = process_audio(file_location)

    # Make predictions using the model
    predicted_emotion, avg_probabilities = predict_and_average(mfcc_segments, model, class_names)

    # Return the predicted emotion and average probabilities as a response
    return {
        "predicted_emotion": predicted_emotion,
        "average_probabilities": avg_probabilities.tolist()  # Convert numpy array to list for JSON serialization
    }
