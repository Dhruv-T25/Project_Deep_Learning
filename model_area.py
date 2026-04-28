import tensorflow as tf
import numpy as np
import video_2_list as v2l
import flow_mod as fm


def give_to_model(frames, model_mode = "N") -> str:
    '''
    This need path of video (mp4, mkv) and by default Normal model is used, if you want to change
    use these characters : "N" for normal model, "P" for pre-trained model  
    '''
    # frames = v2l.video_to_frames(path)  
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
    frames = v2l.video_to_frames("test.mp4")
    result = give_to_model(frames)
    print(result)

