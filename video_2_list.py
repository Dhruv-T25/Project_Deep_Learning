# ⚡ STEP 1: Video Load + Duration Check
import cv2

def load_video(path):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise ValueError("Video not loaded")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    duration = total_frames / fps

    if duration > 30:
        cap.release()
        raise ValueError("Video length > 30 sec (Not allowed)")

    return cap, fps, int(duration)

# STEP 2 🫠 : 1 Frame per Second Extract
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

# expected output : frames = [img1, img2, ..., img30]  max 30

def video_to_frames(path):
    cap, fps, duration = load_video(path)
    frames = extract_frames(cap, fps, duration)
    return frames


def pipeline(path, model):

    frames = video_to_frames(path)   # List[images]
    outputs = model(frames)          # List[str]  ----------- Changes there

    if len(outputs) != len(frames):
        raise ValueError("Model output mismatch")

    return outputs

if __name__ == "__main__":
    pass
    # ## video to img tested ✅
    # video_path = "test.mp4"
    # frames = video_to_frames(video_path)
    # print(len(frames))
    # # print(frames[0])
    # cv2.imshow("Frame 0", frames[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

