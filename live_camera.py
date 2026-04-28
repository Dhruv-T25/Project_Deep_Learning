import cv2
import numpy as np
import tensorflow as tf
import time
import threading
from queue import Queue

MODEL_PATH = "C:\\python pro\\project DL\\Models\\Normal\\emotion_model_N.h5"

emotion_map = {
    0: 'Surprise',
    1: 'Fear',
    2: 'Disgust',
    3: 'Happiness',
    4: 'Sadness',
    5: 'Anger',
    6: 'Neutral'
}

frame_queue = Queue(maxsize=5)
current_emotion = "..."
running = True

model = tf.keras.models.load_model(MODEL_PATH)

# ---------- CAMERA THREAD ----------
def cam_thread():
    global running

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera error")
        running = False
        return

    window_name = "Thread Test"
    cv2.namedWindow(window_name)

    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        small = cv2.resize(frame, (128, 128))

        if not frame_queue.full():
            frame_queue.put(small)

        cv2.putText(frame, f"Emotion: {current_emotion}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        cv2.imshow(window_name, frame)

        # 🔥 key press exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

        # 🔥 cross button detect
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("🪟 Window closed")
            running = False
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------- MODEL THREAD ----------
def model_thread():
    global current_emotion, running

    while running:
        if not frame_queue.empty():
            frame = frame_queue.get()

            try:
                img = frame / 255.0
                img = np.expand_dims(img, axis=0)

                preds = model.predict(img, verbose=0)
                label = np.argmax(preds)

                current_emotion = emotion_map.get(label, "Unknown")

            except Exception as e:
                print("Model error:", e)

        time.sleep(0.05)

# ---------- MAIN ----------
try:
    t1 = threading.Thread(target=cam_thread)
    t2 = threading.Thread(target=model_thread)

    t1.start()
    t2.start()

    while running:
        time.sleep(0.1)

except KeyboardInterrupt:
    print("\n⛔ Ctrl+C detected, stopping...")

    running = False   # 🔥 sab threads band

finally:
    print("🧹 Cleaning up...")

    try:
        t1.join(timeout=1)
        t2.join(timeout=1)
    except:
        pass

    cv2.destroyAllWindows()

    print("✅ Proper exit")