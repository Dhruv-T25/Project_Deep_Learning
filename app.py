# app.py
import streamlit as st
import cv2
import tempfile
import time
import model_area as ma


# ---------- CONFIG ----------
st.set_page_config(page_title="Video Analyzer", layout="centered")

# ---------- TITLE ----------
st.title("🎥 Video Analyzer")
st.write("Upload video (max 30 sec) → process → results")

# ---------- VIDEO UPLOAD ----------
uploaded_file = st.file_uploader(
    "Upload Video",
    type=["mp4", "mkv"]
)

# ---------- FUNCTIONS ----------

def load_video(path):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise ValueError("Video not loaded")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    duration = total_frames / fps

    if duration > 30:
        cap.release()
        raise ValueError("❌ Video > 30 sec not allowed")

    return cap, fps, int(duration)


def extract_frames(cap, fps, duration):
    frames = []

    for sec in range(duration):
        frame_id = int(sec * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

        success, frame = cap.read()
        if not success:
            break

        frames.append(frame)

    cap.release()
    return frames


def video_to_frames(path):
    cap, fps, duration = load_video(path)
    return extract_frames(cap, fps, duration)


# ---------- MODEL PLACEHOLDER ----------
def model_inference(frames):
    # 🔴 tu yaha apna model add karega
    return ["dummy" for _ in frames]


# ---------- MAIN FLOW ----------
if uploaded_file is not None:

    # save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    # show video
    st.video(uploaded_file)

    if st.button("🚀 Process Video"):

        try:
            with st.spinner("Processing..."):
                frames = video_to_frames(tfile.name)

                # st.success(f"✅ Extracted {len(frames)} frames")

                # show first frame
                # st.image(frames[0], caption="First Frame")

                
                # model output
                st.write("## 🧠 Model Thinking")
                output = ma.give_to_model(frames)

                st.write(f"### Model output:{output}")
                
                time.sleep(2)

                # st.write("Extracting features... 🧠")
                # time.sleep(2)

                # st.write("Running deep learning model... 🔥")
                # time.sleep(2)
                

                # st.error("Hum Kuch Nahi Bata Sakte Hum Depression Me Hain 🙂‍↔️🙅‍♂️")
                # st.video("test2.mp4")

        except Exception as e:
            st.error(str(e))