from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import base64
from PIL import Image
from io import BytesIO
import mediapipe as mp
import cv2

app = Flask(__name__)


UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('model/yoga_pose_model.h5')

pose_classes = ['downdog', 'goddess', 'plank', 'tree', 'warrior']

# MediaPipe Pose Detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file_path):
    img = image.load_img(file_path, target_size=(224, 224))  
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 
    return img_array
# Function to draw pose keypoints on image
def draw_pose(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(image_rgb)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    return image

def process_base64_image(data_url):

    header, encoded = data_url.split(',', 1)
    img_data = base64.b64decode(encoded)

    img = Image.open(BytesIO(img_data))
    file_path = 'static/uploads/captured_image.png'
    img.save(file_path)
    return file_path

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    camera_image = request.form.get('camera_image')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        img_array = preprocess_image(file_path)
        predictions = model.predict(img_array)
        predicted_class = pose_classes[np.argmax(predictions)]

        # Draw MediaPipe keypoints on the uploaded image
        image_with_pose = draw_pose(file_path)
        output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'pose_' + filename)
        cv2.imwrite(output_image_path, image_with_pose)

        return render_template('result.html', prediction=predicted_class, image_filename= 'pose_' + filename)

    elif camera_image:
        file_path = process_base64_image(camera_image)
        img_array = preprocess_image(file_path)
        predictions = model.predict(img_array)
        predicted_class = pose_classes[np.argmax(predictions)]

        return render_template('result.html', prediction=predicted_class, image_filename='captured_image.png')

    else:
        return redirect(request.url)

if __name__ == "__main__":
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
