import cv2
import mediapipe as mp

# Fonction de détection des gestes
def detect_gestures(prev_landmarks, curr_landmarks):
    if prev_landmarks and curr_landmarks:
        # Extraction des coordonnées x du pouce et de l'index pour la main droite
        curr_thumb_x = curr_landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP.value]["x"]
        # Extraction des coordonnées y du pouce et de l'index pour la main droite
        curr_thumb_y = curr_landmarks[mp.solutions.hands.HandLandmark.THUMB_TIP.value]["y"]
        print("curr_thumb_x", curr_thumb_x)
        print("Pouce sur Y : ", curr_thumb_y)
        curr_index_y = curr_landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP.value]["y"]
        
        
        curr_index_x = curr_landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP.value]["x"]
        print("curr_index_x", curr_index_x)
        curr_index_mcp_y = curr_landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP.value]["y"]
        print("curr_index_mcp_y", curr_index_mcp_y)
        curr_index_mcp_y = curr_landmarks[mp.solutions.hands.HandLandmark.INDEX_FINGER_PIP.value]["y"]
        print("curr_index_mcp_y", curr_index_mcp_y)
        
        # Extraction des coordonnées x et y du poignet pour la main droite
        curr_wrist_x = curr_landmarks[mp.solutions.hands.HandLandmark.WRIST.value]["x"]
        curr_wrist_y = curr_landmarks[mp.solutions.hands.HandLandmark.WRIST.value]["y"]
        # print("Poiget sur X : ", curr_wrist_x)
        print("Poiget sur Y : ", curr_wrist_y)
        
        # Récupération des coordonnées y des autres doigts pour la main droite
        curr_other_finger_ys = []
        for finger_landmark in [mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
                                mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
                                mp.solutions.hands.HandLandmark.PINKY_TIP]:
            curr_other_finger_ys.append(curr_landmarks[finger_landmark.value]["y"])


        # Détection du geste "Thumbs Up" avec le poing fermé main droite
        if all(curr_thumb_y < finger_y for finger_y in curr_other_finger_ys):
            return "Haut"

        # Détection du geste "Thumbs Down" avec le poing fermé main droite
        if all(curr_thumb_y > finger_y for finger_y in curr_other_finger_ys):
            return "Bas"
        
        # Détection du geste "À gauche" main droite
        if curr_thumb_x < curr_index_x:
            return "Gauche"

        # Détection du geste "À droite" main droite
        if curr_thumb_x > curr_index_x and curr_thumb_y < curr_index_y:
            return "Droite"


# Fonction principale
def main():
    # Initialisation du détecteur de mains MediaPipe
    hands_detector = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

    # Initialisation de la webcam
    camera_video = cv2.VideoCapture(0)
    camera_video.set(3, 1280)
    camera_video.set(4, 720)

    prev_landmarks = None  # Landmarks de la frame précédente

    # Boucle principale
    while camera_video.isOpened():
        # Lecture de l'image depuis la webcam
        read, frame = camera_video.read()
        if not read:
            continue

        # Conversion de l'image en RGB (MediaPipe nécessite une image RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Détection des landmarks des mains dans l'image actuelle
        results = hands_detector.process(frame_rgb)

        # Affichage des landmarks des mains
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dessin des landmarks sur l'image
                mp.solutions.drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=hand_landmarks,
                    connections=mp.solutions.hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )

                # Extraction des landmarks de la main actuelle
                curr_landmarks = [{"x": landmark.x, "y": landmark.y} for landmark in hand_landmarks.landmark]

                # Détection du geste en comparant les landmarks actuels avec ceux de la frame précédente
                gesture = detect_gestures(prev_landmarks, curr_landmarks)
                print(gesture)  # Affichage du geste détecté

                # Stockage des landmarks de la frame actuelle pour la prochaine itération
                prev_landmarks = curr_landmarks

        # Affichage de l'image avec les landmarks et les gestes détectés
        cv2.imshow('Hand Gesture Detection', frame)

        # Interruption de la boucle si la touche 'Esc' est pressée
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    # Libération des ressources
    camera_video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()