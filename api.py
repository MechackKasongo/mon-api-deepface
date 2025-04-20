# app.py

from flask import Flask, request, jsonify
from deepface import DeepFace
import os
import cv2
from PIL import Image
import numpy as np

app = Flask(__name__)

ROOT_DIR = 'dataset'  # ton dossier de référence (avec les sous-dossiers des membres)

def check_image(path):
    img = cv2.imread(path)
    return img is not None

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "Aucune image reçue"}), 400

    file = request.files['image']
    temp_path = "input.jpg"
    file.save(temp_path)

    if not check_image(temp_path):
        return jsonify({"error": "Image illisible"}), 400

    members = {}
    for person_name in os.listdir(ROOT_DIR):
        person_dir = os.path.join(ROOT_DIR, person_name)
        if os.path.isdir(person_dir):
            images = [os.path.join(person_dir, img) for img in os.listdir(person_dir)]
            members[person_name] = images

    for member, images in members.items():
        for image in images:
            try:
                result = DeepFace.verify(temp_path, image, model_name='VGG-Face')
                if result['verified']:
                    confidence = 100 * (1 - result['distance'])
                    return jsonify({
                        "class_name": member,
                        "confidence": round(confidence, 2)
                    })

            except Exception as e:
                print(f"Erreur avec {member} : {e}")

    return jsonify({"class_name": "Inconnu", "confidence": 0})

if __name__ == "__main__":
    app.run(debug=True)
