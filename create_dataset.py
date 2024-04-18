import os  # Importing the os module for file and directory operations
import pickle  # Importing the pickle module for serializing and deserializing Python objects
import mediapipe as mp  # Importing the mediapipe library for hand tracking
import cv2  # Importing OpenCV library for image processing

# Initialize mediapipe components for hand tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'  # Define the directory containing the collected data

data = []  # Initialize a list to store the extracted hand landmark data
labels = []  # Initialize a list to store the corresponding labels (class names)

# Iterate over each subdirectory (class) in the main data directory
for dir_ in os.listdir(DATA_DIR):
    # Iterate over each image file in the current class directory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  # Initialize a list to store the extracted hand landmark data for the current image

        # Process both hands separately
        for hand_index in range(2):
            x_ = []  # Initialize lists to store x-coordinates of hand landmarks
            y_ = []  # Initialize lists to store y-coordinates of hand landmarks

            # Read the image file and convert it to RGB format
            img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Process the image to detect hand landmarks
            results = hands.process(img_rgb)

            # Check if hand landmarks are detected and if the desired hand index exists
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > hand_index:
                hand_landmarks = results.multi_hand_landmarks[hand_index]

                # Iterate over each landmark point in the detected hand
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x  # Get the x-coordinate of the landmark
                    y = hand_landmarks.landmark[i].y  # Get the y-coordinate of the landmark

                    x_.append(x)  # Append x-coordinate to the list
                    y_.append(y)  # Append y-coordinate to the list

                # Normalize landmark coordinates relative to the minimum x and y values
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))  # Append normalized x-coordinate to the data_aux list
                    data_aux.append(y - min(y_))  # Append normalized y-coordinate to the data_aux list

        data.append(data_aux)  # Append the extracted hand landmark data for the current image to the data list
        labels.append(dir_)  # Append the corresponding label (class name) to the labels list

# Save the data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
