# Anti-Sleep Alarm System for Drivers

This project is a prototype driver safety system designed to detect drowsiness using eye tracking with OpenCV and dlib, while enabling real-time alerts using Google Text-to-Speech and Firebase. It can also integrate posture and steering detection via ESP32 sensors for a complete fatigue monitoring system.

---

## 🔍 Key Features
- **Eye Detection**: Calculates Eye Aspect Ratio (EAR) using facial landmarks to detect eye closure.
- **Real-Time Voice Alerts**: Uses Google Text-to-Speech (gTTS) to issue an audio warning when drowsiness is detected.
- **Firebase Integration**: Pushes a "sleep alert" (`sleep: ALERT`) to Firebase Realtime Database.
- **Sensor Integration via ESP32**: Designed to support seat pressure sensors and steering potentiometers.
- **Lightweight Interface**: Uses OpenCV to visualize face and eye tracking with real-time EAR metrics.

---

## 🧠 Technologies Used
- **Python**: OpenCV, dlib, gTTS, Pyrebase
- **ESP32** (for sensor-side integration, future scope)
- **Firebase**: Realtime Database (alerts pushed to cloud)
- **PyCharm**: Python development
- **Camera**: Laptop or USB webcam for live face detection
