import streamlit as st
import cv2
import tempfile
import model_area as ma
import subprocess
import sys

st.set_page_config(page_title="Video Analyzer", layout="centered")

st.title("🎥 Emotion Analyzer")

# ---------- MODE SELECT ----------
mode = st.radio("Select Mode", ["📁 Upload Video", "📷 Live Camera"])

# ---------- VIDEO MODE ----------
if mode == "📁 Upload Video":

    uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mkv"])

    def video_to_frames(path):
        cap = cv2.VideoCapture(path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = int(total_frames / fps)

        if duration > 30:
            cap.release()
            raise ValueError("❌ Video > 30 sec not allowed")

        frames = []
        for sec in range(duration):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        return frames

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        st.video(uploaded_file)

        if st.button("🚀 Process Video"):
            with st.spinner("Processing..."):
                frames = video_to_frames(tfile.name)
                output = ma.give_to_model(frames)

                st.write("## 🧠 Result")
                st.success(output)

# ---------- LIVE CAMERA MODE ----------
elif mode == "📷 Live Camera":

    st.write("Live Emotion Detection (External App)")

    if "proc" not in st.session_state:
        st.session_state.proc = None

    col1, col2 = st.columns(2)

    # ▶️ START
    if col1.button("▶️ Start Camera"):
        if st.session_state.proc is None:
            st.session_state.proc = subprocess.Popen(
                [sys.executable, "live_camera.py"]
            )
            st.success("Camera Started 🚀")
        else:
            st.warning("Already running!")

    # ⛔ STOP
    if col2.button("⛔ Stop Camera"):
        if st.session_state.proc is not None:
            st.session_state.proc.terminate()
            st.session_state.proc = None
            st.success("Camera Stopped ✅")
        else:
            st.warning("Not running!")