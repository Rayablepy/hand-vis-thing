import time
import mediapipe as mp
import cv2 as cv

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def result_output(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print(f"Result: {result}")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='./resources/hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_output)

with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp_ms = int(time.time() * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(mp_image, timestamp_ms=timestamp_ms)

        cv.imshow('window', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()