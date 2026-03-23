import cv2
import os
import numpy as np
import requests
from datetime import datetime

# ================= CONFIG =================
DATASET_PATH = "dataset"
TRAINER_PATH = "trainer/trainer.yml"
LABELS_PATH = "labels.npy"

BOT_TOKEN = "8142685539:AAE6npw9PjR9WY5bM_qnbclwnorEIchkS1A"
CHAT_ID = "1659671016"

os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs("trainer", exist_ok=True)
# ================= CAMERA SETUP =================#
def get_system_camera():
    # Try DirectShow backend (best for Windows)
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if cam.isOpened():
        print("✅ Using system webcam (Camera 0)")
        return cam

    # Fallback (rare case)
    cam = cv2.VideoCapture(0)
    if cam.isOpened():
        print("⚠️ Using fallback webcam (Camera 0)")
        return cam

    print("❌ No system webcam found!")
    return None

# ================= FACE DETECTOR =================
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

recognizer = cv2.face.LBPHFaceRecognizer_create()

# ================= BLUR CHECK =================
def is_blurry(image):
    return cv2.Laplacian(image, cv2.CV_64F).var() < 100

# ================= TELEGRAM ALERT =================
def send_telegram_alert(image_path):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

    with open(image_path, "rb") as img:
        data = {
            "chat_id": CHAT_ID,
            "caption": f"🚨 Intruder Detected!\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        }
        files = {"photo": img}
        requests.post(url, data=data, files=files)

# ================= CAPTURE FACES =================
def capture_faces(name):
    # 🔥 FORCE WEBCAM (0 = laptop webcam)
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cam.isOpened():
        print("❌ Webcam not detected!")
        return

    count = 0
    max_images = 100

    path = os.path.join(DATASET_PATH, name)
    os.makedirs(path, exist_ok=True)

    print("📸 Capturing faces...")

    while True:
        ret, frame = cam.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=6,
            minSize=(80, 80)
        )

        for (x, y, w, h) in faces:

            # ❌ Only allow single face
            if len(faces) != 1:
                continue

            face = gray[y:y+h, x:x+w]

            if is_blurry(face):
                continue

            count += 1
            cv2.imwrite(f"{path}/{count}.jpg", face)

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        # 🔥 DISPLAY IMAGE COUNT (TOP LEFT)
        cv2.putText(
            frame,
            f"Images: {count}/{max_images}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        # 🔥 INSTRUCTION TEXT
        cv2.putText(
            frame,
            "Press ESC to Stop",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow("Face Capture - Webcam", frame)

        # Stop conditions
        if cv2.waitKey(1) == 27 or count >= max_images:
            break

    cam.release()
    cv2.destroyAllWindows()

    print(f"✅ Capture complete! {count} images saved.")

# ================= TRAIN MODEL =================
def train_model():
    print("🔄 Training model...")

    faces = []
    ids = []
    label_map = {}
    current_id = 0

    for person_name in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person_name)

        if not os.path.isdir(person_path):
            continue

        label_map[current_id] = person_name

        for image_name in os.listdir(person_path):
            img_path = os.path.join(person_path, image_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.equalizeHist(img)

            detected = face_detector.detectMultiScale(img, 1.2, 6)

            for (x, y, w, h) in detected:
                face = img[y:y+h, x:x+w]

                if is_blurry(face):
                    continue

                faces.append(face)
                ids.append(current_id)

        current_id += 1

    if len(faces) == 0:
        print("❌ No training data!")
        return

    recognizer.train(faces, np.array(ids))
    recognizer.save(TRAINER_PATH)
    np.save(LABELS_PATH, label_map)

    print("✅ Training complete!")

# ================= RECOGNITION =================
def recognize_faces():
    if not os.path.exists(TRAINER_PATH):
        print("❌ Train model first!")
        return

    recognizer.read(TRAINER_PATH)
    label_map = np.load(LABELS_PATH, allow_pickle=True).item()

    cam = cv2.VideoCapture(0)
    recent_ids = []
    alert_sent = False

    print("🎯 Recognition started...")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(
            gray, 1.2, 6, minSize=(80, 80)
        )

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.equalizeHist(face)

            id_, confidence = recognizer.predict(face)

            recent_ids.append(id_)
            if len(recent_ids) > 5:
                recent_ids.pop(0)

            confirmed = recent_ids.count(id_) >= 3

            if confidence < 60 and confirmed:
                name = label_map.get(id_, "Unknown")
                label = f"{name} ({round(confidence,1)})"
                color = (0,255,0)
                alert_sent = False

            else:
                label = f"Intruder ({round(confidence,1)})"
                color = (0,0,255)

                if not alert_sent:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    img_path = f"intruder_{timestamp}.jpg"

                    cv2.imwrite(img_path, frame)
                    print("🚨 Intruder detected! Sending alert...")

                    send_telegram_alert(img_path)
                    alert_sent = True

            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, label, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("AI Security System", frame)

        if cv2.waitKey(1) == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

# ================= MAIN MENU =================
if __name__ == "__main__":
    while True:
        print("\n===== AI SECURITY SYSTEM =====")
        print("1. Capture Faces")
        print("2. Train Model")
        print("3. Start Recognition")
        print("4. Exit")

        choice = input("Enter choice: ")

        if choice == "1":
            name = input("Enter name: ")
            capture_faces(name)

        elif choice == "2":
            train_model()

        elif choice == "3":
            recognize_faces()

        elif choice == "4":
            break