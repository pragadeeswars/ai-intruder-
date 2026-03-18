import cv2
import os
import numpy as np

dataset_path = "dataset"
trainer_path = "trainer/trainer.yml"

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

faces = []
labels = []
label_map = {}
current_id = 0

for person_name in os.listdir(dataset_path):

    person_path = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_path):
        continue

    label_map[current_id] = person_name

    for image_name in os.listdir(person_path):

        image_path = os.path.join(person_path, image_name)

        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces_detected = face_detector.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces_detected:

            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face,(200,200))

            faces.append(face)
            labels.append(current_id)

    current_id += 1


recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,
    neighbors=8,
    grid_x=8,
    grid_y=8
)

recognizer.train(faces,np.array(labels))

if not os.path.exists("trainer"):
    os.makedirs("trainer")

recognizer.save(trainer_path)

print("Training complete!")
print("Labels:",label_map)