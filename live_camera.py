import cv2
import time
import model_area as ma

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not working")
    exit()

last_time = 0
current_emotion = "..."
buffer = []   # frames collect karne ke liye

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        try:
            # ---- har 0.5 sec pe frame collect ----
            if time.time() - last_time > 0.5:
                last_time = time.time()
                buffer.append(frame)

            # ---- batch ready (2–4 frames) ----
            if len(buffer) >= 3:
                current_emotion = ma.give_to_model(buffer)  # 🔥 direct call
                buffer.clear()  # reset

        except Exception as e:
            print(f"⚠️ Model error: {e}")

        # ---- display ----
        cv2.putText(frame, f"Emotion: {current_emotion}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        cv2.imshow("Live Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"🔥 Critical Error: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Clean exit")