import time
import mediapipe as mp
import numpy as np
import cv2 as cv

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_styles = mp.tasks.vision.drawing_styles
#styles below and drawframe function extracted from mediapipe colab notebook
margin = 10
font_size = 1
font_thickness = 1
txt_colour = (255,0,0)
latest_result = None

def drawframe(currframe, result: HandLandmarkerResult | None):
    if result is None:
        return currframe

    landmarks_list = result.hand_landmarks
    handedness_list = result.handedness
    mod_frame = np.copy(currframe)

    for i in range(len(landmarks_list)):
        landmarks = landmarks_list[i]
        handedness = handedness_list[i]

        mp_drawing.draw_landmarks(
            mod_frame,
            landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style()
        )
        height, width, _ = mod_frame.shape
        x_coords = [landmark.x for landmark in landmarks]
        y_coords = [landmark.y for landmark in landmarks]
        txt_x = int(min(x_coords)*width)
        txt_y = int(min(y_coords)*height) - margin

        cv.putText(mod_frame,f"{handedness[0].category_name}",(txt_x,txt_y),cv.FONT_HERSHEY_DUPLEX,font_size,txt_colour,font_thickness,cv.LINE_AA)
    return mod_frame

def result_output(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='./resources/hand_landmarker.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_output)
with HandLandmarker.create_from_options(options) as landmarker:
    cap = cv.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("System error")
            break

        timestamp_ms = int(time.time() * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv.cvtColor(frame,cv.COLOR_BGR2RGB))
        landmarker.detect_async(mp_image, timestamp_ms=timestamp_ms)
        annotated = drawframe(frame.copy(), latest_result)
        cv.imshow('window', annotated)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
