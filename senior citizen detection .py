import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import datetime
import csv
import os
from mtcnn import MTCNN

# --- Load pre-trained models ---
age_model = tf.keras.models.load_model("saved_age_gender_models/AGE_GROUP_MODELS.h5")
gender_model = tf.keras.models.load_model("saved_age_gender_models/GENDER_MODELS.h5")

GENDER_LABELS = ['Male', 'Female']
AGE_LABELS = ['0-18', '19-30', '31-45', '46-60', '61+']

# --- Face detector ---
detector = MTCNN()

# --- CSV logging setup ---
csv_file = "visits_log.csv"
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Approx Age", "Age Group", "Gender", "Senior Citizen", "Time"])

# --- GUI setup ---
root = tk.Tk()
root.title("Senior Citizen Identification")
root.geometry("800x600")
root.configure(bg="#f0f0f0")

image_label = Label(root)
image_label.pack(pady=20)

result_text = tk.StringVar()
result_label = Label(root, textvariable=result_text, font=("Arial", 14), bg="#f0f0f0")
result_label.pack(pady=10)

# --- Predict Function ---
def predict_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    img = cv2.imread(file_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img_rgb)

    result_string = ""

    for face in faces:
        x, y, w, h = face['box']
        face_img = img[y:y+h, x:x+w]
        face_resized = cv2.resize(face_img, (96, 96))
        face_array = face_resized.astype("float32") / 255.0
        face_array = np.expand_dims(face_array, axis=0)

        # --- Predict Age ---
        age_pred = age_model.predict(face_array)[0]
        approx_age = int(age_pred.flatten()[0] * 100)
  

        # --- Determine Age Group ---
        if approx_age <= 18:
            age_group = AGE_LABELS[0]
        elif approx_age <= 30:
            age_group = AGE_LABELS[1]
        elif approx_age <= 45:
            age_group = AGE_LABELS[2]
        elif approx_age <= 60:
            age_group = AGE_LABELS[3]
        else:
            age_group = AGE_LABELS[4]

        # --- Predict Gender ---
        gender_pred = gender_model.predict(face_array)[0]
        gender_idx = np.argmax(gender_pred)
        gender_label = GENDER_LABELS[gender_idx]

        is_senior = "Yes" if approx_age >= 60 else "No"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # --- Save to CSV ---
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([approx_age, age_group, gender_label, is_senior, timestamp])

        result_string += f"Age: {approx_age} ({age_group}), Gender: {gender_label}, Senior: {is_senior}\n"

        # --- Draw on image ---
        label = f"{gender_label}, {approx_age}"
        color = (0, 255, 0) if is_senior == "Yes" else (255, 0, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if not faces:
        result_string = "No faces detected."

    # --- Show image in GUI ---
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil = img_pil.resize((500, 400))
    imgtk = ImageTk.PhotoImage(image=img_pil)
    image_label.configure(image=imgtk)
    image_label.image = imgtk

    result_text.set(result_string)

# --- Upload Button ---
upload_btn = Button(root, text="Upload & Predict", command=predict_image,
                    font=("Arial", 12), bg="#4CAF50", fg="white", padx=10, pady=5)
upload_btn.pack(pady=20)

root.mainloop()
