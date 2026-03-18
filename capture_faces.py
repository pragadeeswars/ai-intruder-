import cv2
import os

name = input("Enter person's name: ")

dataset_path = f"dataset/{name}"
os.makedirs(dataset_path, exist_ok=True)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)

count = 0
max_images =40

print("Press SPACE to capture")
print("Press Q to quit")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.putText(frame,f"Captured {count}/{max_images}",(10,30),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("Capture Faces",frame)

    key = cv2.waitKey(1)

    if key == 32:

        if len(faces) > 0:

            x,y,w,h = faces[0]

            face = gray[y:y+h, x:x+w]

            face = cv2.resize(face,(200,200))

            face = cv2.equalizeHist(face)

            count += 1

            cv2.imwrite(f"{dataset_path}/{count}.jpg", face)

            print("Captured",count)

    if key == ord('q'):
        break

    if count >= max_images:
        break

cap.release()
cv2.destroyAllWindows()