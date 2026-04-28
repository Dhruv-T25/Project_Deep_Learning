import cv2
import numpy as np
import tensorflow as tf
import time

# ---------- LOAD MODEL ----------
model = tf.keras.models.load_model("C:\\python pro\\project DL\\Models\\Normal\\emotion_model_N.h5")

emotion_map = {
    0: 'Surprise',
    1: 'Fear',
    2: 'Disgust',
    3: 'Happiness',
    4: 'Sadness',
    5: 'Anger',
    6: 'Neutral'
}

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not working")
    exit()

last_time = 0
current_emotion = "..."

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame not received")
            continue   # break nahi → loop chalne de

        try:
            # ---- process every 0.5 sec ----
            if time.time() - last_time > 0.5:
                last_time = time.time()

                img = cv2.resize(frame, (224, 224))
                img = img / 255.0
                img = np.expand_dims(img, axis=0)

                preds = model.predict(img, verbose=0)
                label = np.argmax(preds)

                current_emotion = emotion_map.get(label, "Unknown")

        except Exception as e:
            print(f"⚠️ Frame processing error: {e}")
            # continue → next frame pe try karega

        # ---- display ----
        cv2.putText(frame, f"Emotion: {current_emotion}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        cv2.imshow("Live Emotion Detection", frame)

        # exit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Exiting...")
            break

except KeyboardInterrupt:
    print("⛔ Manually stopped (Ctrl+C)")

except Exception as e:
    print(f"🔥 Critical Error: {e}")

finally:
    # ---------- CLEANUP ----------
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Camera released & windows closed")