import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Charger le modèle entraîné
model = load_model('model.h5')

# Définir les noms des gestes
# gesture_names = ['back', 'down', 'flip', 'forward', 'left', 'right', 'stop', 'up']
gesture_names = ['down','forward', 'up', 'null']

# Paramètres pour la capture vidéo
camera_video = cv2.VideoCapture(0)
camera_video.set(3, 640)  # Largeur de la vidéo
camera_video.set(4, 480)  # Hauteur de la vidéo

while True:
    # Capturer une image de la caméra
    read, frame = camera_video.read()
    if not read:
        continue

    # Prétraiter l'image pour la mettre dans le format attendu par le modèle
    resized_frame = cv2.resize(frame, (150, 150))  # Redimensionner l'image
    resized_frame = resized_frame / 255.0  # Normaliser les valeurs des pixels (entre 0 et 1)
    input_image = np.expand_dims(resized_frame, axis=0)  # Ajouter une dimension pour correspondre à la forme attendue par le modèle

    # Prédire les gestes à partir de l'image
    predictions = model.predict(input_image)
    gesture_index = np.argmax(predictions)
    probability = predictions[0][gesture_index]
    print(gesture_index)
    if gesture_index == 3 and probability > 0.5:
        gesture_name = "null"
    else:
        gesture_name = gesture_names[gesture_index]

    # Afficher le nom du geste sur l'image
    cv2.putText(frame, f'Gesture: {gesture_name}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Afficher l'image avec le résultat
    cv2.imshow('Hand Gesture Detection', frame)

    # Quitter la boucle si la touche 'Esc' est pressée
    key = cv2.waitKey(1)
    if key == 27:
        break

# Libérer les ressources
camera_video.release()
cv2.destroyAllWindows()
