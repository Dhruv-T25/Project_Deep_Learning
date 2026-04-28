import streamlit as st
import cv2
import tempfile
import model_area as ma
import time

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

    st.write("Click Start to begin live detection")

    start = st.button("▶️ Start Camera")
    stop = st.button("⛔ Stop")

    frame_placeholder = st.empty()

    if start:
        cap = cv2.VideoCapture(0)
        last_time = 0
        emotion = "..."

        try:
            while cap.isOpened():

                ret, frame = cap.read()
                if not ret:
                    continue

                # ---- inference every 0.5 sec ----
                if time.time() - last_time > 0.5:
                    last_time = time.time()

                    try:
                        emotion = ma.give_to_model([frame])  # 🔥 reuse function
                    except:
                        emotion = "Error"

                # overlay
                cv2.putText(frame, f"Emotion: {emotion}",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2)

                # show in streamlit
                frame_placeholder.image(frame, channels="BGR")

                # stop condition
                if stop:
                    break

        finally:
            cap.release()