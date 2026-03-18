import cv2
import os
from datetime import datetime

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer/trainer.yml")

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

names = ["pragadeeswar","sachin"]

cap = cv2.VideoCapture(0)

print("AI Intruder System Started")

while True:

    ret,frame = cap.read()

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:

        face = gray[y:y+h,x:x+w]
        face = cv2.resize(face,(200,200))

        id,confidence = recognizer.predict(face)

        if confidence < 70:
            name = names[id]
            color = (0,255,0)
            label = f"{name} {round(100-confidence)}%"

        else:
            label = "INTRUDER"
            color = (0,0,255)

            with open("logs/intruder_log.txt","a") as f:
                f.write(f"Intruder detected at {datetime.now()}\n")

        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        cv2.putText(frame,label,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

    cv2.imshow("AI Intruder System",frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()