import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'B', 2: 'C'}
while True:

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        # Initialize data_aux for both hands
        data_aux_left = []
        data_aux_right = []

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # Calculate bounding box coordinates for each hand
            x_min, y_min = W, H
            x_max, y_max = 0, 0
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * W), int(landmark.y * H)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            # Extract features from each hand
            hand_data_aux = []
            for landmark in hand_landmarks.landmark:
                # Normalize landmark coordinates relative to the bounding box of the hand
                x_norm = (landmark.x * W - x_min) / (x_max - x_min)
                y_norm = (landmark.y * H - y_min) / (y_max - y_min)
                hand_data_aux.append(x_norm)
                hand_data_aux.append(y_norm)

            # Pad or truncate data_aux to contain 42 features (21 per hand)
            hand_data_aux += [0] * (42 - len(hand_data_aux))

            # Append features to the corresponding hand's data_aux list
            if x_min < W / 2:  # Left hand
                data_aux_left.extend(hand_data_aux)
            else:  # Right hand
                data_aux_right.extend(hand_data_aux)

        # Ensure each hand's feature vector has exactly 42 features
        data_aux_left += [0] * (42 - len(data_aux_left))
        data_aux_right += [0] * (42 - len(data_aux_right))

        # Combine features from both hands and pad/truncate to 84 features
        combined_data_aux = data_aux_left + data_aux_right
        combined_data_aux += [0] * (84 - len(combined_data_aux))

        # Display predicted text
        text_position = (50, 50)
        prediction = model.predict([np.asarray(combined_data_aux)])
        predicted_character = labels_dict[int(prediction[0])]
        cv2.putText(frame, predicted_character, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()