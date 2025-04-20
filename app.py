from flask import Flask, request, jsonify
from deepface import DeepFace
import os

app = Flask(__name__)

# Dossier dataset avec les sous-dossiers de visages
DATASET_PATH = "dataset"

@app.route("/", methods=["GET"])
def index():
    return "API DeepFace OK!"

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "Aucune image reçue"}), 400

    image = request.files['image']
    image_path = os.path.join("temp.jpg")
    image.save(image_path)

    members = {}
    for person_name in os.listdir(DATASET_PATH):
        person_dir = os.path.join(DATASET_PATH, person_name)
        if os.path.isdir(person_dir):
            member_images = [os.path.join(person_dir, img) for img in os.listdir(person_dir)]
            members[person_name] = member_images

    for member, images in members.items():
        for img in images:
            try:
                result = DeepFace.verify(image_path, img, model_name='VGG-Face')
                if result['verified']:
                    confidence = 100 * (1 - result['distance'])
                    return jsonify({
                        "class": member,
                        "confidence": f"{confidence:.2f}%"
                    })
            except Exception as e:
                continue

    return jsonify({"class": "Inconnu", "confidence": "0%"})
