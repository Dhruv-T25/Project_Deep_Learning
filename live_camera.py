import cv2
import numpy as np
import tensorflow as tf
import time

def live_emotion_detector(model_path):

    model = tf.keras.models.load_model(model_path)

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
        raise RuntimeError("Camera not working")

    last_time = 0
    current_emotion = "..."

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # ---- fast inference (0.5 sec gap) ----
            if time.time() - last_time > 0.5:
                last_time = time.time()

                img = cv2.resize(frame, (224, 224))
                img = img / 255.0
                img = np.expand_dims(img, axis=0)

                preds = model.predict(img, verbose=0)
                label = np.argmax(preds)

                current_emotion = emotion_map.get(label, "Unknown")

            # yield frame + emotion
            yield frame, current_emotion

    finally:
        cap.release()

if __name__ == "__main__":
    model_path = "C:\\python pro\\project DL\\Models\\Normal\\emotion_model_N.h5"

    try:
        for frame, emotion in live_emotion_detector(model_path):

            print(f"Detected Emotion: {emotion}")

            # display frame
            cv2.putText(frame, f"Emotion: {emotion}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)

            cv2.imshow("Test Camera", frame)

            # exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")

    finally:
        cv2.destroyAllWindows()
