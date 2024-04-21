import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(dir_path):
        continue  # Skip non-directory items
    for img_file in os.listdir(dir_path):
        data_aux = []
        img_path = os.path.join(dir_path, img_file)

        # Read the image file and handle errors
        try:
            img = cv2.imread(img_path)
            if img is None:
                print(f"Warning: Unable to read image file {img_path}. Skipping.")
                continue
        except Exception as e:
            print(f"Error reading image file {img_path}: {e}")
            continue

        # Convert the image to RGB format
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = hands.process(img_rgb)

        for hand_index in range(2):
            x_ = []
            y_ = []

            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > hand_index:
                hand_landmarks = results.multi_hand_landmarks[hand_index]

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

        data.append(data_aux)
        labels.append(dir_)

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
