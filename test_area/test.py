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

