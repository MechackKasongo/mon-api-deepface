from flask import Flask, request, jsonify
from deepface import DeepFace
import os
import cv2
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Dossier contenant les images d'entraînement (dataset)
root_dir = "dataset"  # Ce dossier doit être ajouté dans ton projet

def check_image(path):
    img = cv2.imread(path)
    return img is not None

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'Aucune image envoyée'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Fichier vide'}), 400

    filename = secure_filename(file.filename)
    input_path = os.path.join("uploads", filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(input_path)

    # Chargement des images de référence
    for person in os.listdir(root_dir):
        person_dir = os.path.join(root_dir, person)
        if os.path.isdir(person_dir):
            for image_file in os.listdir(person_dir):
                reference_path = os.path.join(person_dir, image_file)
                try:
                    result = DeepFace.verify(input_path, reference_path, model_name='VGG-Face')
                    if result["verified"]:
                        return jsonify({
                            'personne': person,
                            'confiance': f"{100 * (1 - result['distance']):.2f} %"
                        })
                except Exception as e:
                    print(e)
                    continue

    return jsonify({'message': "Personne non reconnue"}), 200

if __name__ == '__main__':
    app.run()
