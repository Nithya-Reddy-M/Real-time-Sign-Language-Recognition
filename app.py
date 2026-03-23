import streamlit as st
import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# --- PAGE CONFIG ---
st.set_page_config(page_title="ASL Detector", layout="centered")

st.title("🤟 Sign Language Detection")
st.write("Real-time sign recognition using LSTM")

# --- CONTROLS ---
start = st.button("Start Camera")
stop = st.button("Stop Camera")

# --- LOAD MODEL ---
model = load_model("asl_action_model.keras")

# --- LOAD LABELS ---
DATA_PATH = "ASL_Data"
actions = np.array(sorted(os.listdir(DATA_PATH)))

# --- UI ---
frame_window = st.image([])
prediction_text = st.empty()

# --- CAMERA ---
if start:
    cap = cv2.VideoCapture(0)
    sequence = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_window.image(frame, channels="BGR")

        hand_results, pose_results = mediapipe_detection(frame)
        keypoints = extract_keypoints(hand_results, pose_results)

        sequence.append(keypoints)
        sequence = sequence[-30:]

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            prediction = actions[np.argmax(res)]

            prediction_text.write(f"Prediction: {prediction}")

        if stop:
            break

    cap.release()