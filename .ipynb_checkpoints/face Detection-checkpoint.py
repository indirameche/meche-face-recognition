import cv2
import os
import numpy as np
from PIL import Image

dataset_dir = 'dataset'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)


cam = cv2.VideoCapture(0)
cam.set(3, 640)  # Set video width
cam.set(4, 480)  # Set video height


face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


while True:
    face_id = input('\nEnter user ID (numeric only) and press <return>: ')
    if face_id.isdigit():
        face_id = int(face_id)
        break
    else:
        print("Please enter a valid numeric ID.")

print("\n[INFO] Initializing face capture. Look at the camera and wait...")
count = 0


while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)  
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        count += 1

  
        cv2.imwrite(f"{dataset_dir}/User.{face_id}.{count}.jpg", gray[y:y + h, x:x + w])

        if count % 10 == 0:
            print(f"[INFO] Captured {count} images for User ID: {face_id}")

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
    if k == 27:  # Exit on ESC key
        print("[INFO] Exiting program.")
        break
    elif count >= 30:  # Take 30 face samples and stop video
        print(f"[INFO] Captured {count} images. Stopping.")
        break


print("\n[INFO] Exiting Program and cleaning up.")
cam.release()
cv2.destroyAllWindows()


def getImagesAndLabels(path):
    if not os.path.exists(path):
        raise ValueError(f"The specified path does not exist: {path}")


    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not imagePaths:
        raise ValueError("No images found in the specified directory.")

    face_samples = []
    ids = []
    
    for imagePath in imagePaths:
        try:
            PIL_img = Image.open(imagePath).convert('L') 
            img_numpy = np.array(PIL_img, 'uint8')
            id = int(os.path.split(imagePath)[1].split(".")[1])
            
            faces = face_detector.detectMultiScale(img_numpy)
            if len(faces) == 0:
                print(f"No faces found in image: {imagePath}")
                continue 

            for (x, y, w, h) in faces:
                face_samples.append(img_numpy[y:y+h, x:x+w])
                ids.append(id)

        except Exception as e:
            print(f"Error processing image {imagePath}: {e}")
    
    if not face_samples:
        raise ValueError("No face samples extracted. Check your dataset.")

    return face_samples, ids


trainer_dir = 'trainer'
if not os.path.exists(trainer_dir):
    os.makedirs(trainer_dir)


recognizer = cv2.face.LBPHFaceRecognizer_create()


face_samples, ids = getImagesAndLabels(dataset_dir)


recognizer.train(face_samples, np.array(ids))
recognizer.save(f'{trainer_dir}/trainer.yml')

print("Model trained and saved as trainer.yml.")
