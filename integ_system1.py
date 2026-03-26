import cv2
import os
import numpy as np
import requests
import time
from datetime import datetime
from ultralytics import YOLO
from deepface import DeepFace

# ================= CONFIG =================
DATASET_PATH = "dataset"
EMBEDDINGS_PATH = "embeddings.npy"

BOT_TOKEN = "8142685539:AAE6npw9PjR9WY5bM_qnbclwnorEIchkS1A"
CHAT_ID = "1659671016"

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

ALERT_COOLDOWN = 15

os.makedirs(DATASET_PATH, exist_ok=True)

# ================= YOLO MODEL =================
yolo_model = YOLO("yolov8n.pt")

# ================= FACE DETECTOR =================
face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ================= CAMERA =================
def get_camera():
    cam = cv2.VideoCapture(CAMERA_INDEX)

    if not cam.isOpened():
        for i in range(3):
            cam = cv2.VideoCapture(i)
            if cam.isOpened():
                print(f"✅ Using Camera {i}")
                break

    cam.set(3, FRAME_WIDTH)
    cam.set(4, FRAME_HEIGHT)
    return cam

# ================= TELEGRAM =================
def send_telegram_alert(image_path):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

    with open(image_path, "rb") as img:
        data = {
            "chat_id": CHAT_ID,
            "caption": f"🚨 Intruder Detected!\nTime: {datetime.now()}"
        }
        files = {"photo": img}
        requests.post(url, data=data, files=files)

# ================= YOLO HUMAN DETECTION =================
def detect_humans(frame):
    results = yolo_model(frame, conf=0.6)
    boxes = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            if cls != 0 or conf < 0.6:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            boxes.append((x1, y1, x2-x1, y2-y1))

    return boxes

# ================= FACE CAPTURE =================
def capture_faces(name):
    cam = get_camera()
    path = os.path.join(DATASET_PATH, name)
    os.makedirs(path, exist_ok=True)

    count = 0
    max_images = 40
    last_capture = 0

    print("📸 Capturing faces... Move your head")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=25)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_detector.detectMultiScale(gray, 1.2, 6)

        for (x, y, w, h) in faces:
            if len(faces) != 1:
                continue

            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))

            if time.time() - last_capture > 0.4:
                count += 1
                cv2.imwrite(f"{path}/{count}.jpg", face)
                last_capture = time.time()

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

        cv2.putText(frame, f"Images: {count}/{max_images}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        cv2.imshow("Capture Faces", frame)

        if cv2.waitKey(1) == 27 or count >= max_images:
            break

    cam.release()
    cv2.destroyAllWindows()
    print("✅ Capture done")

# ================= BUILD EMBEDDINGS =================
def build_embeddings():
    database = {}

    print("🔄 Building embeddings...")

    for person in os.listdir(DATASET_PATH):
        path = os.path.join(DATASET_PATH, person)

        if not os.path.isdir(path):
            continue

        embeddings = []

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            try:
                emb = DeepFace.represent(
                    img_path,
                    model_name="ArcFace",  # 🔥 better model
                    detector_backend="opencv",
                    enforce_detection=True
                )[0]["embedding"]

                embeddings.append(emb)
            except:
                continue

        if embeddings:
            database[person] = np.mean(embeddings, axis=0)

    np.save(EMBEDDINGS_PATH, database)
    print("✅ Embeddings built successfully")

# ================= FACE MATCH =================
def find_match(face_img, database):
    try:
        emb = DeepFace.represent(
            face_img,
            model_name="ArcFace",
            detector_backend="opencv",
            enforce_detection=True
        )[0]["embedding"]

        min_dist = float("inf")
        identity = "Unknown"

        for person in database:
            dist = np.linalg.norm(np.array(emb) - np.array(database[person]))

            if dist < min_dist:
                min_dist = dist
                identity = person

        # 🔥 STRICT LOGIC
        if min_dist < 0.7:
            return identity, min_dist
        elif min_dist < 1.0:
            return "Unknown", min_dist
        else:
            return "Intruder", min_dist

    except:
        return "Error", 999

# ================= TRACKING =================
tracker = {}
track_id = 0

def assign_id(x, y):
    global track_id

    for tid in tracker:
        tx, ty = tracker[tid]
        if abs(x - tx) < 50 and abs(y - ty) < 50:
            tracker[tid] = (x, y)
            return tid

    tracker[track_id] = (x, y)
    track_id += 1
    return track_id - 1

# ================= RECOGNITION =================
def recognize_faces():
    if not os.path.exists(EMBEDDINGS_PATH):
        print("❌ Build embeddings first")
        return

    database = np.load(EMBEDDINGS_PATH, allow_pickle=True).item()

    cam = get_camera()
    last_alert_time = 0
    alerted_ids = set()

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)

        humans = detect_humans(frame)

        for (x,y,w,h) in humans:
            person_img = frame[y:y+h, x:x+w]

            gray = cv2.cvtColor(person_img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.2, 5)

            name = "No Face"
            distance = 999

            # 🔥 ONLY 1 FACE → VALID
            if len(faces) == 1:
                (fx, fy, fw, fh) = faces[0]

                face_img = person_img[fy:fy+fh, fx:fx+fw]
                face_img = cv2.resize(face_img, (224,224))

                name, distance = find_match(face_img, database)

            tid = assign_id(x, y)

            color = (0,255,0) if name not in ["Intruder", "Unknown"] else (0,0,255)

            label = f"{name} | ID:{tid} | {round(distance,2)}"

            # 🔥 ALERT ONLY TRUE INTRUDER
            if name == "Intruder" and tid not in alerted_ids:
                if time.time() - last_alert_time > ALERT_COOLDOWN:
                    img_path = f"intruder_{int(time.time())}.jpg"
                    cv2.imwrite(img_path, frame)
                    send_telegram_alert(img_path)

                    alerted_ids.add(tid)
                    last_alert_time = time.time()

            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
            cv2.putText(frame, label, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if len(tracker) > 50:
            tracker.clear()

        cv2.imshow("AI SECURITY SYSTEM", frame)

        if cv2.waitKey(1) == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

# ================= MAIN =================
if __name__ == "__main__":
    while True:
        print("\n===== AI SECURITY SYSTEM =====")
        print("1. Capture Faces")
        print("2. Build Embeddings")
        print("3. Start Recognition")
        print("4. Exit")

        c = input("Choice: ")

        if c == "1":
            name = input("Enter name: ")
            capture_faces(name)

        elif c == "2":
            build_embeddings()

        elif c == "3":
            recognize_faces()

        elif c == "4":
            break