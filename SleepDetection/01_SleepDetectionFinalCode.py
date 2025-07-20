import cv2
import dlib
from scipy.spatial import distance
from gtts import gTTS
import os
import time
import pyrebase
import tempfile

# -------------------------------------- Firebase Initialization -----------------------------------------------------

firebaseConfig = {
  "apiKey": "AIzaSyCLU2w5XcrPmMYDCGxlSIZou55tv1nGfHQ",
  "authDomain": "dispmeddem.firebaseapp.com",
  "databaseURL": "https://dispmeddem-default-rtdb.firebaseio.com",
  "projectId": "dispmeddem",
  "storageBucket": "dispmeddem.firebasestorage.app",
  "messagingSenderId": "77501478469",
  "appId": "1:77501478469:web:50c261c14b7ac27f6f952c",
  "measurementId": "G-KW3CVHRJ10"
}
firebase = pyrebase.initialize_app(firebaseConfig)
db = firebase.database()

# -------------------------------------- Voice Notification Setup --------------------------------------------------

def play_audio_message(message):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_filename = temp_audio.name
    tts = gTTS(text=message, lang='en', slow=False)
    tts.save(temp_filename)
    os.system(f"start {temp_filename}")  # Windows

# -------------------------------------- EAR Calculation Function ---------------------------------------------------
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A + B) / (2.0 * C)
    return ear_aspect_ratio

# -------------------------------------- Firebase Update Function ---------------------------------------------------
def firebase_update_sleep():
    data = {"sleep": "ALERT"}
    db.child("KL05S6628").set(data)

# -------------------------------------- Sleep Detection Setup ------------------------------------------------------

cap = cv2.VideoCapture(0)  # Change to the appropriate camera index or RTSP URL
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Drowsiness detection parameters
EAR_THRESHOLD = 0.18  # Adjust this based on testing
FRAME_LIMIT = 20  # Number of consecutive frames required to confirm sleep
frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if the camera is not providing frames

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = hog_face_detector(gray)

    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)
        leftEye = []
        rightEye = []

        for n in range(36, 42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            leftEye.append((x, y))
            next_point = 36 if n == 41 else n + 1
            x2, y2 = face_landmarks.part(next_point).x, face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        for n in range(42, 48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            rightEye.append((x, y))
            next_point = 42 if n == 47 else n + 1
            x2, y2 = face_landmarks.part(next_point).x, face_landmarks.part(next_point).y
            cv2.line(frame, (x, y), (x2, y2), (0, 255, 0), 1)

        left_ear = calculate_EAR(leftEye)
        right_ear = calculate_EAR(rightEye)
        EAR = round((left_ear + right_ear) / 2, 2)

        if EAR < EAR_THRESHOLD:
            frame_counter += 1  # Increase frame count if EAR remains low
        else:
            frame_counter = 0  # Reset if eyes are open

        if frame_counter >= FRAME_LIMIT:  # Trigger alert after sustained low EAR
            cv2.putText(frame, "DROWSY ALERT!", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            cv2.putText(frame, "Wake up!", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            print("Drowsy Alert!")
            firebase_update_sleep()
            play_audio_message("Wake up! You are feeling sleepy.")
            frame_counter = 0  # Reset after alert

        print(f"EAR: {EAR}, Frame Counter: {frame_counter}")

    cv2.imshow("Driver Drowsiness Detection", frame)

    key = cv2.waitKey(1)
    if key == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
