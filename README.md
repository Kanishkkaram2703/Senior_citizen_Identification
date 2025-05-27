# Senior Citizen Identification

## 1. Abstract

This project implements a deep learning-based system that can detect and classify people from video or webcam feeds. It predicts age and gender of each person, and if a person is 60 years or older, it marks them as a senior citizen. Detected age, gender, and timestamp are saved into a CSV file for further analysis. This system is especially useful for malls, stores, or public places to track senior citizens and demographic data.

## 2. Introduction

### Problem Statement:
Malls and public places often require systems to analyze foot traffic, identify senior citizens for priority service, and log visit details. Manual tracking is inefficient and error-prone. This project aims to automate this process using AI.

### Objectives:

- Predict age and gender of people in real-time from video/webcam feed.
- Mark individuals aged 60+ as senior citizens.
- Store age, gender, and time of visit in an Excel/CSV file.
- Provide optional GUI interface for ease of use.

### Applications:

- Malls and retail environments for customer profiling.
- Queue management with senior citizen prioritization.
- Public health and demographic tracking.

## 3. Literature Review

- **Face Detection with MTCNN**: Efficient for detecting multiple faces in real-time.
- **Age and Gender Classification with CNNs**: Convolutional Neural Networks are effective for predicting continuous variables like age and categorical ones like gender.
- **Logging with CSV**: Simple and scalable way to store user data and timestamps.

## 4. Methodology

### Data Used:
- **UTKFace** or **IMDB-WIKI** dataset for training the model.
- Dataset includes face images with labeled age and gender.

### Preprocessing:
- Resize and normalize face images.
- Convert to grayscale or RGB depending on model input.
- Label encode gender (0: Male, 1: Female) and treat age as numeric.

### Model Training:
- CNN architecture with Conv2D, MaxPooling, Flatten, Dense.
- Two output heads: one for age (regression), one for gender (classification).
- Optimizer: Adam
- Losses: MSE for age, Binary Crossentropy for gender
- Evaluation: MAE (age), Accuracy (gender)

### Detection Flow:
1. Capture frame from webcam/video.
2. Detect all faces using MTCNN.
3. Predict age and gender for each face.
4. If age ≥ 60, label as "Senior Citizen".
5. Draw bounding box and label.
6. Log details (age, gender, timestamp) in CSV.

## 5. Implementation

### Core Modules:
- `main.py`: Face detection, age & gender prediction, CSV logging.
- `model_training.ipynb`: Model building and training notebook.
- `utils.py`: Preprocessing and utility functions.

### Sample Logging Code:

```python
with open("visitor_log.csv", mode="a", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([predicted_age, predicted_gender, datetime.now()])
```

Prediction Output Format:
Age: 67 (Senior)

Gender: Male

Logged at: 2025-05-27 10:23:45

**6. Results**
Age MAE: ~6.5 years

Gender Accuracy: ~90%

Real-Time Speed: ~5–10 FPS on CPU

Include a few screenshots of GUI or console output.

**7. Challenges and Limitations**
Challenges:
Varying lighting and camera quality.

Real-time multi-face detection latency.

Accurate age prediction for children/senior faces.

**Limitations**:
No GUI included by default (optional).

Limited to frontal face images.

Model misclassifies in low-light or blurry inputs.

**8. Conclusion**
The system effectively detects and logs senior citizens in real-time from video or webcam feeds. This solution can assist businesses, malls, and public systems in managing crowd demographics and offering services to the elderly with care.

🔄** Pre-trained Model**
You can skip training by downloading the pretrained model from this Google Drive link:
📦 Google Drive - Age & Gender Model

🚀** Features**
Real-time face detection using MTCNN

Age and gender prediction via CNN

Automatic senior citizen identification (age ≥ 60)

CSV logging of age, gender, and timestamp

Optional GUI with image/video feed display

📁** Dataset**
Dataset used: UTKFace or IMDB-WIKI (not included in repo)

Download from Kaggle and place in the dataset/ folder.

**✅ Requirements**
Python 3.8+

OpenCV

NumPy

TensorFlow

MTCNN

Pillow

Scikit-learn

Pandas

Matplotlib (for training plots)

**💾 Installation**

git clone https://github.com/Kanishkkaram2703/Senior_citizen_Identification.git
cd Senior_citizen_Identification
pip install -r requirements.txt
🧠 Model Training
To train your own model:

Download and extract the dataset into dataset/ folder.

Open model_training.ipynb and run all cells.

The model will be saved as .keras.

OR, download the ready-to-use model from the Google Drive link.

Pretrained model not hosted on GitHub (due to size).

## 🔄 Pre-trained Model

Download the trained `.keras` model directly from this Google Drive folder:  
📦 [Google Drive - Senior Citizen Identification](https://drive.google.com/drive/folders/1qsjD8CMuT5IU3eQ2XtIxaIE2TESI6iet?usp=drive_link)



