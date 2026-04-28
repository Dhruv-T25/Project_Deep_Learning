# word warp (alt + z) to make code more readable

import tensorflow as tf
import numpy as np
import video_2_list as v2l
import flow_mod as fm

video_path = "test.mp4"
frames = v2l.video_to_frames(video_path)   # list of images (numpy arrays)

# preprocess function 
def process(image):
    image = tf.image.resize(image, (256, 256)) # resize to model input size 
    image = tf.cast(image / 255.0, tf.float32)
    return image

processed_frames = [process(frame) for frame in frames] # Model can run this without pre-processing (tested)

# convert to tensor (VERY IMPORTANT)
processed_frames = tf.stack(processed_frames) 


# model inference 
# modelP = tf.keras.models.load_model("C:\\python pro\\project DL\\Models\\pre_trained\\emotion_model.h5") # here P means pre-trained model (tested)
modelN = tf.keras.models.load_model("C:\\python pro\\project DL\\Models\\Normal\\emotion_model_N.h5") # here N means normal model (tested)

emotion_map = {
    0: 'Surprise',
    1: 'Fear',
    2: 'Disgust',
    3: 'Happiness',
    4: 'Sadness',
    5: 'Anger',
    6: 'Neutral'
}

predictions = modelN.predict(processed_frames)

pred_labels = np.argmax(predictions, axis=1)

final_output = [emotion_map[i] for i in pred_labels]

print(fm.get_mode(final_output))


'''
import tensorflow as tf
import numpy as np
import video_2_list as v2l
import flow_mod as fm


def give_to_model(path : str, model_mode = "N") -> str:
    ## this is fucntion help line
    # This need path of video (mp4, mkv) and by default Normal model is used, if you want to change
    # use these characters : "N" for normal model, "P" for pre-trained model  
    
    frames = v2l.video_to_frames(path)  
    processed_frames = [frame for frame in frames] 
    processed_frames = tf.stack(processed_frames)
    if model_mode == "N": 
        model = tf.keras.models.load_model(
            "C:\\python pro\\project DL\\Models\\Normal\\emotion_model_N.h5"
            ) # here N means normal model (tested)
    elif model_mode == "P":
        model = tf.keras.models.load_model(
            "C:\\python pro\\project DL\\Models\\pre_trained\\emotion_model.h5"
            ) # here P means pre-trained model (tested)
        pass

    emotion_map = {
        0: 'Surprise',
        1: 'Fear',
        2: 'Disgust',
        3: 'Happiness',
        4: 'Sadness',
        5: 'Anger',
        6: 'Neutral'
    }

    predictions = model.predict(processed_frames)
    pred_labels = np.argmax(predictions, axis=1)
    final_output = [emotion_map[i] for i in pred_labels]
    return fm.get_mode(final_output)

if __name__ == "__main__":
    result = give_to_model("test.mp4")
    print(result)
'''



'''
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
'''
